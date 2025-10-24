import numpy as np
import streamlit as st
import pickle
import re

class ManualBernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha 
        self.prior_probs = None
        self.conditional_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        X_binary = (X > 0).astype(int)

        self.prior_probs = np.zeros(n_classes)
        self.conditional_probs = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self._classes):
            X_c = X_binary[y == c]
            self.prior_probs[idx] = X_c.shape[0] / n_samples
            N_iC = np.sum(X_c, axis=0) 
            N_C = X_c.shape[0]
            self.conditional_probs[idx] = (N_iC + self.alpha) / (N_C + 2 * self.alpha)

    def predict(self, X):
        X_binary = (X > 0).astype(int)
        predictions = [self._predict(x) for x in X_binary]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        epsilon = 1e-9 
        for idx, c in enumerate(self._classes):
            prior = np.log(self.prior_probs[idx])
            prob_feature_present = self.conditional_probs[idx] 
            prob_feature_absent = 1.0 - prob_feature_present 
            
            log_likelihood = np.sum(
                x * np.log(prob_feature_present + epsilon) + 
                (1 - x) * np.log(prob_feature_absent + epsilon)
            )
            posterior = prior + log_likelihood
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    
    def predict_proba(self, X):
        X_binary = (X > 0).astype(int)
        probs = []
        epsilon = 1e-9
        for x in X_binary:
            class_probs = []
            for idx, _ in enumerate(self._classes):
                prior = np.log(self.prior_probs[idx])
                prob_feature_present = self.conditional_probs[idx]
                prob_feature_absent = 1.0 - prob_feature_present

                log_likelihood = np.sum(
                    x * np.log(prob_feature_present + epsilon) +
                    (1 - x) * np.log(prob_feature_absent + epsilon)
                )
                posterior = prior + log_likelihood
                class_probs.append(np.exp(posterior))
            class_probs = np.array(class_probs)
            probs.append(class_probs / np.sum(class_probs))
        return np.array(probs)

# =============================
# 1 Cấu hình trang
# =============================
st.set_page_config(page_title="Phân loại Email Spam/Ham", layout="centered")

st.markdown("""
<style>
div.block-container { padding-top: 1rem; }
.stButton>button { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

st.title(" Ứng dụng Phân loại Email: SPAM vs HAM")
st.write("Nhập nội dung email vào ô bên dưới để phân loại.")

# =============================
# 2 Hàm xử lý văn bản
# =============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # xóa link
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', text) # xóa ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================
# 3 Tải mô hình và TF-IDF
# =============================
@st.cache_resource
def load_artifacts(model_path='final_model.pkl', tfidf_path='tfidf_vectorizer.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf, None
    except Exception as e:
        return None, None, e

model, tfidf, load_error = load_artifacts()
if load_error:
    st.error(f"Lỗi khi tải mô hình hoặc vectorizer: {load_error}")
    st.stop()
else:
    st.success(" Mô hình và TF-IDF đã được tải thành công!")

# =============================
# 4 Sidebar hướng dẫn
# =============================
with st.sidebar:
    st.header(" Hướng dẫn")
    st.write("- Dán nội dung email vào ô chính.")
    st.write("- Nhấn nút **Phân loại** để nhận kết quả.")
    st.write("- Ví dụ mẫu có thể dùng để thử:")

    if st.button("Chèn ví dụ SPAM"):
        st.session_state['email_input'] = (
            'Congratulations! You have won $10,000! Click the link below to claim your prize now!')
    if st.button("Chèn ví dụ HAM"):
        st.session_state['email_input'] = (
            'Hello, I would like to confirm our meeting with you tomorrow afternoon.')

# =============================
# 5 Khu vực nhập email
# =============================
if 'email_input' not in st.session_state:
    st.session_state['email_input'] = ''

email_input = st.text_area(" Nội dung email", value=st.session_state['email_input'], height=200)

col1, col2 = st.columns([3, 1])
with col2:
    classify = st.button("Phân loại")

# =============================
# 6 Xử lý phân loại
# =============================
if classify:
    if email_input.strip() == "":
        st.warning(" Vui lòng nhập nội dung email trước khi phân loại.")
    else:
        try:
            with st.spinner(' Đang phân loại...'):
                processed_text = preprocess_text(email_input)
                email_vec = tfidf.transform([processed_text]).toarray()
                result = model.predict(email_vec)[0]

                # Lấy xác suất nếu có
                try:
                    proba = model.predict_proba(email_vec)[0]
                except Exception:
                    proba = None

            # =============================
            #  Kết luận đúng theo mô hình SPAM=1, HAM=0
            # =============================
            if result == 1:
                st.error(" Kết luận: SPAM (email rác / lừa đảo)")
            else:
                st.success(" Kết luận: HAM (email bình thường)")

            # Hiển thị xác suất
            if proba is not None:
                spam_prob = float(proba[1]) if len(proba) > 1 else 0.0
                st.info(f"Xác suất SPAM: {spam_prob * 100:.2f}%")

            # Hiển thị chi tiết dự đoán
            with st.expander(" Chi tiết dự đoán (raw output)"):
                st.write({
                    'processed_text': processed_text,
                    'predicted_label': int(result),
                    'probabilities': proba.tolist() if proba is not None else None
                })

        except Exception as e:
            st.error(f" Lỗi khi dự đoán: {e}")
