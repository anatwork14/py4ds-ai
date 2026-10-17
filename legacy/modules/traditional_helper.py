import cv2
import numpy as np
import pandas as pd
import time
import concurrent.futures # Added for threading

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

class BasePreprocessor:
    def __init__(self, img_size=(150, 150)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_label_encoder_fitted = False

    def _clean(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        img = cv2.equalizeHist(img)
        return cv2.resize(img, self.img_size)

    def extract_features_single(self, img):
        raise NotImplementedError("Subclasses must implement extract_features_single")

    def _process_row(self, row):
        # Helper method to map over threads
        img = self._clean(row['filepath'])
        if img is not None:
            feat = self.extract_features_single(img)
            return feat, row['class']
        return None, None

    def extract_features(self, df, split_name, is_training=False):
        features, labels = [], []
        subset = df[df['split'] == split_name]

        print(f"Extracting features for {split_name} split ({len(subset)} images)...")
        
        # OpenCV releases the GIL, making ThreadPoolExecutor highly effective
        with concurrent.futures.ThreadPoolExecutor() as executor:
            rows = [row for _, row in subset.iterrows()]
            results = list(executor.map(self._process_row, rows))

        # Unpack the threaded results
        for feat, label in results:
            if feat is not None:
                features.append(feat)
                labels.append(label)

        X = np.array(features)
        
        # Encode labels first
        if is_training or not self.is_label_encoder_fitted:
            y = self.label_encoder.fit_transform(labels)
            self.is_label_encoder_fitted = True
        else:
            y = self.label_encoder.transform(labels)

        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

class HOGPreprocessor(BasePreprocessor):
    def __init__(self, img_size=(150, 150)):
        super().__init__(img_size)
    
    def extract_features_single(self, img):
        # Using 144x144 window for 150x150 images
        hog = cv2.HOGDescriptor(_winSize=(144,144), _blockSize=(16,16),
                                _blockStride=(8,8), _cellSize=(8,8), _nbins=9)
        return hog.compute(img).flatten()

class SIFTPreprocessor(BasePreprocessor):
    def __init__(self, img_size=(150, 150), vocab_size=100):
        super().__init__(img_size)
        self.vocab_size = vocab_size
        self.kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, n_init=3)
        self.sift = cv2.SIFT_create()

    def build_vocabulary_from_df(self, df, max_samples=1000):
        print(f"Building SIFT vocabulary (size={self.vocab_size})...")
        descriptors_list = []
        train_subset = df[df['split'] == 'train']
        
        sample_size = min(max_samples, len(train_subset))
        train_subset = train_subset.sample(sample_size, random_state=42)

        for _, row in train_subset.iterrows():
            img = self._clean(row['filepath'])
            if img is not None:
                _, des = self.sift.detectAndCompute(img, None)
                if des is not None:
                    descriptors_list.append(des)

        if not descriptors_list:
            raise ValueError("No SIFT descriptors found. Check image paths.")

        all_descriptors = np.vstack(descriptors_list)
        self.kmeans.fit(all_descriptors)
        print(f"Vocabulary of size {self.vocab_size} built successfully.")

    def extract_features_single(self, img):
        _, des = self.sift.detectAndCompute(img, None)
        histogram = np.zeros(self.vocab_size)
        if des is not None:
            predictions = self.kmeans.predict(des)
            for p in predictions:
                histogram[p] += 1

        # L2 Normalization
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
        return histogram

def run_traditional_pipeline(df, vocab_size=100, feature_configs=["hog", "sift"], max_sift_samples=1500, pca_components=500, show_report=False):
    results = []
    predictions_dict = {}
    class_names = []

    for f_type in feature_configs:
        print(f"\n========================================")
        print(f" FEATURE EXTRACTION: {f_type.upper()} ")
        print(f"========================================")

        if f_type == "hog":
            preprocessor = HOGPreprocessor()
        elif f_type == "sift":
            preprocessor = SIFTPreprocessor(vocab_size=vocab_size)
            preprocessor.build_vocabulary_from_df(df, max_samples=max_sift_samples)
        else:
            continue

        # 2. Extract and Scale
        X_train_scaled, y_train = preprocessor.extract_features(df, "train", is_training=True)
        X_test_scaled, y_test = preprocessor.extract_features(df, "test", is_training=False)

        # 3. Dimensionality Reduction
        if f_type == "hog":
            print(f"Applying PCA (n_components={pca_components}) to {f_type.upper()}...")
            # svd_solver='randomized' massively reduces time complexity
            pca = PCA(n_components=pca_components, whiten=True, svd_solver='randomized', random_state=42)
            X_train_final = pca.fit_transform(X_train_scaled)
            X_test_final = pca.transform(X_test_scaled)
        else:
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled

        # 4. Model Training
        models = {
            "SVM": LinearSVC(dual=False, C=0.01, max_iter=2000, random_state=42),
            "LR": LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs", random_state=42)
        }

        for m_name, model in models.items():
            exp_name = f"{f_type.upper()} + {m_name}"
            print(f"-> Running {exp_name}...")

            model.fit(X_train_final, y_train)

            y_pred = model.predict(X_test_final)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            class_names = [str(c) for c in preprocessor.label_encoder.classes_]
            predictions_dict[exp_name] = (y_test, y_pred)

            if show_report:
                report = classification_report(y_test, y_pred, target_names=class_names)
                print(f"Detailed Report for {exp_name}:")
                print(report)

            results.append({
                "Combination": exp_name,
                "Accuracy": acc,
                "F1 Score": f1,
                # "Train Time (s)": round(elapsed, 3)
            })

    results_df = pd.DataFrame(results)
    return results_df, predictions_dict, class_names

