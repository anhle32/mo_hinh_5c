# =============================================================================
# ỨNG DỤNG STREAMLIT: DỰ BÁO RỦI RO TÍN DỤNG KHÁCH HÀNG THEO MÔ HÌNH 5C
# Tái hiện pipeline từ notebook: Logistic Regression dự báo biến PD
# (0 = không có rủi ro, 1 = có rủi ro) từ 24 biến đầu vào thang Likert 1-5
# thuộc 5 nhóm: TC (Tư cách), NL (Năng lực), DK (Điều kiện), V (Vốn), TS (Tài sản).
# =============================================================================

import streamlit as st

# ---- 1) set_page_config PHẢI là lệnh Streamlit đầu tiên ----
st.set_page_config(
    layout="wide",
    page_title="Dự báo rủi ro tín dụng 5C",
    page_icon="🏦",
)

# ---- 2) Import & các hàm cache dùng chung ----
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Tập biến LẤY ĐÚNG TỪ NOTEBOOK
FEATURES = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET = "PD"
CHART_HEIGHT = 340  # chiều cao cố định cho biểu đồ để lưới cân đối


@st.cache_data(show_spinner="Đang nạp dữ liệu...")
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Hàm nạp dữ liệu DÙNG CHUNG (nhận bytes để hashable với cache_data).

    Notebook không tạo biến phái sinh, không xử lý thiếu, không scaler —
    nên hàm chỉ đọc CSV và trả về DataFrame nguyên trạng.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Trả về danh sách cột còn thiếu so với yêu cầu."""
    return [c for c in required if c not in df.columns]


def pick_chart(df: pd.DataFrame, col: str):
    """Tự chọn loại biểu đồ theo kiểu dữ liệu thực tế của biến."""
    s = df[col].dropna()
    if col == TARGET:
        # Mục tiêu phân loại nhị phân → bar phân phối lớp
        vc = s.value_counts().sort_index()
        labels = vc.index.map({0: "0 - Không rủi ro", 1: "1 - Có rủi ro"}).fillna(vc.index.astype(str))
        fig = px.bar(
            x=labels, y=vc.values,
            labels={"x": col, "y": "Số quan sát"},
            title=f"Phân phối lớp biến mục tiêu {col}",
            color=labels,
        )
        fig.update_layout(showlegend=False)
    elif pd.api.types.is_numeric_dtype(s):
        nunique = s.nunique()
        if nunique <= 10:
            # Số rời rạc (thang Likert 1-5) → bar theo value_counts
            vc = s.value_counts().sort_index()
            fig = px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": f"Mức đánh giá {col}", "y": "Số quan sát"},
                title=f"Phân phối biến {col}",
            )
        else:
            # Số liên tục → histogram
            fig = px.histogram(s, x=col, title=f"Phân phối biến {col}")
    else:
        vc = s.astype(str).value_counts().head(20)
        fig = px.bar(
            x=vc.index, y=vc.values,
            labels={"x": col, "y": "Số quan sát"},
            title=f"Phân phối biến {col}",
        )
    fig.update_layout(height=CHART_HEIGHT, margin=dict(t=50, b=10))
    return fig


# =============================================================================
# ---- 3) THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH ----
# =============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải tệp dữ liệu (.csv)",
        type=["csv"],
        help="Tệp CSV chứa 24 biến đầu vào (TC1–TC5, NL1–NL4, DK1–DK5, V1–V6, TS1–TS4) và biến mục tiêu PD.",
    )

    # Notebook chỉ dùng MỘT mô hình (LogisticRegression) → không có selectbox chọn mô hình
    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test_size)",
        min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        help="Tỷ lệ dữ liệu dành cho tập kiểm tra. Notebook gốc dùng 0.2 (20%).",
    )
    random_state = st.number_input(
        "Hạt giống ngẫu nhiên (random_state)",
        min_value=0, max_value=9999, value=23, step=1,
        help="Cố định cách chia train/test để tái lập kết quả. Notebook gốc dùng 23.",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        C_value = st.slider(
            "Hệ số điều chuẩn C",
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help="Nghịch đảo cường độ regularization. Notebook dùng mặc định C=1.0.",
        )
        max_iter = st.number_input(
            "Số vòng lặp tối đa (max_iter)",
            min_value=100, max_value=5000, value=100, step=100,
            help="Số vòng lặp tối đa của thuật toán tối ưu. Mặc định sklearn là 100.",
        )

    st.divider()
    run_train = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
        help="Chia dữ liệu, huấn luyện Logistic Regression và tính các chỉ tiêu kiểm định.",
    )

# =============================================================================
# ---- 4) THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG ----
# =============================================================================
st.title("🏦 Dự báo rủi ro tín dụng khách hàng theo mô hình 5C")
st.caption(
    "Ứng dụng huấn luyện mô hình **Logistic Regression** dự báo khả năng khách hàng "
    "có rủi ro tín dụng (PD = 1) hay không (PD = 0) dựa trên 24 tiêu chí đánh giá "
    "thang Likert 1–5 thuộc 5 nhóm: Tư cách (TC), Năng lực (NL), Điều kiện (DK), "
    "Vốn (V) và Tài sản đảm bảo (TS). Đầu vào kỳ vọng: tệp CSV có cấu trúc như tệp mẫu `5c.csv`."
)

if uploaded_file is None:
    st.info(
        "👈 Vui lòng tải tệp dữ liệu **.csv** ở thanh bên trái để bắt đầu. "
        "Tệp cần chứa 24 cột biến đầu vào (TC1–TC5, NL1–NL4, DK1–DK5, V1–V6, TS1–TS4) "
        "và cột mục tiêu **PD**."
    )
    st.stop()

# Nạp dữ liệu qua hàm cache dùng chung — có xử lý lỗi định dạng
try:
    file_bytes = uploaded_file.getvalue()
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"❌ Không đọc được tệp CSV. Vui lòng kiểm tra định dạng tệp. Chi tiết lỗi: {e}")
    st.stop()

if df.empty:
    st.error("❌ Tệp dữ liệu rỗng, không có quan sát nào.")
    st.stop()

missing_cols = validate_columns(df, FEATURES + [TARGET])
if missing_cols:
    st.error(
        "❌ Tệp dữ liệu thiếu các cột bắt buộc sau: **" + ", ".join(missing_cols) + "**. "
        "Vui lòng kiểm tra lại cấu trúc tệp."
    )
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(
    f"Tóm tắt nhanh: **{df.shape[0]:,}** quan sát × **{df.shape[1]}** cột · "
    f"Mô hình sử dụng **{len(FEATURES)}** biến đầu vào và biến mục tiêu **{TARGET}**."
)
st.divider()

# =============================================================================
# ---- 5) KHỐI HUẤN LUYỆN — chạy DUY NHẤT khi bấm nút, lưu session_state ----
# =============================================================================
if run_train:
    try:
        X = df[FEATURES]
        y = df[TARGET]

        if y.nunique() < 2:
            st.error("❌ Biến mục tiêu PD chỉ có một lớp duy nhất, không thể huấn luyện mô hình phân loại.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(C=C_value, max_iter=int(max_iter))
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yproba_test = model.predict_proba(X_test)[:, 1]

        results_df = pd.DataFrame(
            {
                "y_test (thực tế)": y_test.values,
                "yhat_test (dự báo)": yhat_test,
                "Xác suất có rủi ro": yproba_test.round(4),
            },
            index=y_test.index,
        )

        # Notebook không có scaler/encoder → bộ tiền xử lý là None (giữ đúng pipeline gốc)
        st.session_state["model"] = model
        st.session_state["preprocessor"] = None
        st.session_state["results_df"] = results_df
        st.session_state["eval"] = {
            "y_test": y_test.values,
            "yhat_test": yhat_test,
            "yproba_test": yproba_test,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "params": {
                "test_size": test_size,
                "random_state": int(random_state),
                "C": C_value,
                "max_iter": int(max_iter),
            },
        }
        st.toast("✅ Huấn luyện mô hình thành công!", icon="🎉")
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình huấn luyện: {e}")
        st.stop()

# =============================================================================
# ---- 6) CÁC TAB NỘI DUNG ----
# =============================================================================
tab_overview, tab_viz, tab_eval, tab_predict = st.tabs(
    [
        "📊 Tổng quan dữ liệu",
        "📈 Trực quan hóa dữ liệu",
        "🧪 Kết quả huấn luyện & kiểm định",
        "🎯 Sử dụng mô hình",
    ]
)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# -----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng tệp", f"{uploaded_file.size / (1024 * 1024):.3f} MB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=320):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả các biến của mô hình")
    st.caption("Chỉ mô tả 24 biến đầu vào và biến mục tiêu PD được đưa vào mô hình.")
    st.dataframe(df[FEATURES + [TARGET]].describe(), use_container_width=True)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# -----------------------------------------------------------------------------
with tab_viz:
    model_vars = [TARGET] + FEATURES  # biến mục tiêu ưu tiên trước
    default_vars = [TARGET, "TC1", "NL1", "TS1"]  # 4 biến ưu tiên: y trước, rồi các biến đầu vào

    selected_vars = st.multiselect(
        "Chọn biến cần trực quan hóa (mặc định 4 biến ưu tiên, biến mục tiêu PD trước)",
        options=model_vars,
        default=default_vars,
        help="Có 24 biến đầu vào nên ứng dụng cho phép chọn linh hoạt; biểu đồ tự chọn loại theo kiểu dữ liệu.",
    )

    if not selected_vars:
        st.info("Vui lòng chọn ít nhất một biến để vẽ biểu đồ.")
    else:
        # Lưới 2 cột cân đối, chiều cao cố định
        for i in range(0, len(selected_vars), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(selected_vars[i : i + 2]):
                with cols[j]:
                    st.plotly_chart(
                        pick_chart(df, col_name),
                        use_container_width=True,
                        key=f"viz_{col_name}",
                    )

# -----------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# -----------------------------------------------------------------------------
with tab_eval:
    if "model" not in st.session_state:
        st.info("⏳ Mô hình chưa được huấn luyện. Vui lòng bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        ev = st.session_state["eval"]
        y_test = ev["y_test"]
        yhat = ev["yhat_test"]
        yproba = ev["yproba_test"]

        st.caption(
            f"Cấu hình đã huấn luyện: test_size = {ev['params']['test_size']}, "
            f"random_state = {ev['params']['random_state']}, C = {ev['params']['C']}, "
            f"max_iter = {ev['params']['max_iter']} · "
            f"Tập huấn luyện: {ev['n_train']} quan sát · Tập kiểm tra: {ev['n_test']} quan sát."
        )

        # Chỉ tiêu vô hướng — phân loại có giám sát
        acc = accuracy_score(y_test, yhat)
        prec = precision_score(y_test, yhat, zero_division=0)
        rec = recall_score(y_test, yhat, zero_division=0)
        f1 = f1_score(y_test, yhat, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yproba)
        except ValueError:
            auc = None

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.4f}", help="Tỷ lệ dự báo đúng trên tập kiểm tra (tương ứng model.score trong notebook).")
        m2.metric("Precision", f"{prec:.4f}")
        m3.metric("Recall", f"{rec:.4f}")
        m4.metric("F1-score", f"{f1:.4f}")
        m5.metric("ROC-AUC", f"{auc:.4f}" if auc is not None else "N/A")

        st.divider()
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, yhat)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Dự báo: 0", "Dự báo: 1"],
                y=["Thực tế: 0", "Thực tế: 1"],
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_cm.update_layout(height=CHART_HEIGHT, coloraxis_showscale=False, margin=dict(t=20, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.subheader("Đường cong ROC")
            if auc is not None:
                fpr, tpr, _ = roc_curve(y_test, yproba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {auc:.4f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(dash="dash")))
                fig_roc.update_layout(
                    height=CHART_HEIGHT,
                    xaxis_title="Tỷ lệ dương tính giả (FPR)",
                    yaxis_title="Tỷ lệ dương tính thật (TPR)",
                    margin=dict(t=20, b=10),
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Không tính được ROC-AUC (tập kiểm tra chỉ có một lớp).")

        st.subheader("Báo cáo phân loại (classification report)")
        report = classification_report(y_test, yhat, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

        st.subheader("Bảng so sánh dự báo và thực tế trên tập kiểm tra")
        st.caption("Tái hiện bảng `{'yhat_test', 'y_test'}` trong notebook, bổ sung xác suất có rủi ro.")
        with st.container(height=320):
            st.dataframe(st.session_state["results_df"], use_container_width=True)

# -----------------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# -----------------------------------------------------------------------------
with tab_predict:
    if "model" not in st.session_state:
        st.info("⏳ Mô hình chưa được huấn luyện. Vui lòng bấm nút **🚀 Huấn luyện mô hình** ở thanh bên trái.")
    else:
        model = st.session_state["model"]
        # Notebook không dùng scaler/encoder → dữ liệu mới đưa thẳng vào mô hình (đúng pipeline gốc)

        mode = st.radio(
            "Chọn chế độ sử dụng mô hình",
            ["📝 Nhập trực tiếp một khách hàng", "📂 Tải tệp dữ liệu theo cấu trúc X_test"],
            horizontal=True,
        )

        # ---------------- CHẾ ĐỘ 1: NHẬP TRỰC TIẾP ----------------
        if mode == "📝 Nhập trực tiếp một khách hàng":
            st.caption(
                "Nhập mức đánh giá (thang 1–5) cho 24 tiêu chí của khách hàng cần thẩm định. "
                "Giá trị mặc định là trung vị của dữ liệu; khoảng nhập giới hạn theo min–max thực tế của dữ liệu."
            )
            group_labels = {
                "TC": "Tư cách khách hàng (TC)",
                "NL": "Năng lực khách hàng (NL)",
                "DK": "Điều kiện (DK)",
                "V": "Vốn (V)",
                "TS": "Tài sản đảm bảo (TS)",
            }
            with st.form("form_predict_single"):
                inputs = {}
                for prefix, label in group_labels.items():
                    group_feats = [f for f in FEATURES if f.rstrip("0123456789") == prefix]
                    st.markdown(f"**{label}**")
                    cols = st.columns(len(group_feats))
                    for k, feat in enumerate(group_feats):
                        with cols[k]:
                            inputs[feat] = st.number_input(
                                feat,
                                min_value=int(df[feat].min()),
                                max_value=int(df[feat].max()),
                                value=int(df[feat].median()),
                                step=1,
                                help=f"Mức đánh giá tiêu chí {feat} (thang Likert).",
                            )
                submitted = st.form_submit_button("🎯 Dự báo", type="primary", use_container_width=True)

            if submitted:
                X_new = pd.DataFrame([inputs])[FEATURES]
                pred = int(model.predict(X_new)[0])
                proba = model.predict_proba(X_new)[0]

                r1, r2, r3 = st.columns(3)
                r1.metric("Kết quả dự báo (PD)", pred)
                r2.metric("Xác suất KHÔNG có rủi ro", f"{proba[0] * 100:.2f}%")
                r3.metric("Xác suất CÓ rủi ro", f"{proba[1] * 100:.2f}%")

                if pred == 1:
                    st.error(f"⚠️ Khách hàng được dự báo **CÓ RỦI RO TÍN DỤNG** (xác suất {proba[1] * 100:.2f}%).")
                else:
                    st.success(f"✅ Khách hàng được dự báo **KHÔNG CÓ RỦI RO TÍN DỤNG** (xác suất {proba[0] * 100:.2f}%).")

        # ---------------- CHẾ ĐỘ 2: DỰ BÁO HÀNG LOẠT TỪ FILE ----------------
        else:
            st.caption(
                "Tải tệp CSV chứa **đúng 24 cột biến đầu vào**: "
                + ", ".join(FEATURES)
                + ". Các cột khác (nếu có) sẽ được giữ lại trong bảng kết quả nhưng không đưa vào mô hình."
            )
            batch_file = st.file_uploader("Tải tệp dữ liệu cần dự báo (.csv)", type=["csv"], key="batch_uploader")

            if batch_file is not None:
                try:
                    df_new = pd.read_csv(io.BytesIO(batch_file.getvalue()))
                except Exception as e:
                    st.error(f"❌ Không đọc được tệp CSV. Chi tiết lỗi: {e}")
                    st.stop()

                if df_new.empty:
                    st.error("❌ Tệp dữ liệu rỗng.")
                else:
                    missing = validate_columns(df_new, FEATURES)
                    if missing:
                        st.error("❌ Tệp thiếu các cột bắt buộc sau: **" + ", ".join(missing) + "**.")
                    else:
                        try:
                            X_batch = df_new[FEATURES]
                            preds = model.predict(X_batch)
                            probas = model.predict_proba(X_batch)

                            out = df_new.copy()
                            out["PD_du_bao"] = preds
                            out["Xac_suat_khong_rui_ro"] = (probas[:, 0] * 100).round(2)
                            out["Xac_suat_co_rui_ro"] = (probas[:, 1] * 100).round(2)

                            n_risk = int((preds == 1).sum())
                            b1, b2, b3 = st.columns(3)
                            b1.metric("Số quan sát dự báo", f"{len(out):,}")
                            b2.metric("Số KH có rủi ro (PD=1)", f"{n_risk:,}")
                            b3.metric("Tỷ lệ có rủi ro", f"{n_risk / len(out) * 100:.2f}%")

                            st.subheader("Bảng kết quả dự báo")
                            with st.container(height=360):
                                st.dataframe(out, use_container_width=True)

                            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "⬇️ Tải kết quả dự báo (CSV)",
                                data=csv_bytes,
                                file_name="ket_qua_du_bao_PD.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"❌ Lỗi khi dự báo hàng loạt: {e}")
