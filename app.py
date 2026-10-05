import pandas as pd
import streamlit as st
import plotly.express as px

from predict import analyze_feedback


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Feedback AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        background: #f8fafc;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Student Feedback AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Student Feedback Sentiment, Emotion & Aspect Analyzer'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Student Feedback AI")

    st.divider()

    selected_page = st.radio(
        "Choose a module",
        [
            "🔍 Analyze Feedback",
            "📊 Batch Analysis",
            "📈 Dashboard",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.markdown(
        """
        ### AI Architecture

        **CNN + Bidirectional LSTM**

        ### AI Outputs

        💭 Sentiment

        ❤️ Emotion

        🏫 Multiple Aspects

        ### Analysis

        **Hybrid CNN-BiLSTM + NLP Rules**
        """
    )


# ============================================================
# SINGLE FEEDBACK ANALYSIS
# ============================================================

if selected_page == "🔍 Analyze Feedback":

    st.header("🔍 Analyze Student Feedback")

    st.write(
        "Enter student feedback. The system analyzes "
        "overall sentiment, emotion and multiple "
        "college-related aspects."
    )

    st.divider()

    feedback = st.text_area(
        "Student Feedback",
        placeholder=(
            "Example: The classrooms are comfortable, "
            "but the internet is very slow. "
            "The faculty members are supportive."
        ),
        height=160
    )

    analyze_button = st.button(
        "🔍 Analyze Feedback",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not feedback.strip():

            st.warning(
                "Please enter some feedback."
            )

        else:

            with st.spinner(
                "AI is analyzing the feedback..."
            ):

                result = analyze_feedback(
                    feedback
                )

            st.success(
                "Analysis completed successfully."
            )

            st.divider()

            # ------------------------------------------------
            # OVERALL RESULTS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            # Sentiment
            with col1:

                st.subheader(
                    "💭 Overall Sentiment"
                )

                sentiment = result[
                    "sentiment"
                ]

                if sentiment == "Positive":

                    st.success(
                        sentiment
                    )

                elif sentiment == "Negative":

                    st.error(
                        sentiment
                    )

                elif sentiment == "Mixed":

                    st.warning(
                        sentiment
                    )

                else:

                    st.info(
                        sentiment
                    )

                st.caption(
                    "Confidence: "
                    + f"{result['sentiment_confidence']:.2%}"
                )

                st.progress(
                    min(
                        max(
                            float(
                                result[
                                    "sentiment_confidence"
                                ]
                            ),
                            0.0
                        ),
                        1.0
                    )
                )

            # Emotion
            with col2:

                st.subheader(
                    "❤️ Dominant Emotion"
                )

                st.info(
                    result["emotion"]
                )

                st.caption(
                    "Confidence: "
                    + f"{result['emotion_confidence']:.2%}"
                )

                st.progress(
                    min(
                        max(
                            float(
                                result[
                                    "emotion_confidence"
                                ]
                            ),
                            0.0
                        ),
                        1.0
                    )
                )

            # Primary aspect
            with col3:

                st.subheader(
                    "🏫 Primary Aspect"
                )

                st.warning(
                    result["aspect"]
                )

                st.caption(
                    "Confidence: "
                    + f"{result['aspect_confidence']:.2%}"
                )

                st.progress(
                    min(
                        max(
                            float(
                                result[
                                    "aspect_confidence"
                                ]
                            ),
                            0.0
                        ),
                        1.0
                    )
                )

            st.divider()

            # ------------------------------------------------
            # SUBMITTED FEEDBACK
            # ------------------------------------------------

            st.subheader(
                "📝 Submitted Feedback"
            )

            st.info(
                feedback
            )

            # ------------------------------------------------
            # ASPECT LEVEL ANALYSIS
            # ------------------------------------------------

            st.subheader(
                "🔎 Aspect-Level Analysis"
            )

            aspect_details = result[
                "aspect_details"
            ]

            if aspect_details:

                aspect_rows = []

                for detail in aspect_details:

                    aspect_rows.append({

                        "Aspect":
                            detail["aspect"],

                        "Sentiment":
                            detail["sentiment"],

                        "Confidence":
                            f"{detail['confidence']:.2%}",

                        "Positive":
                            detail["positive_count"],

                        "Negative":
                            detail["negative_count"],

                        "Neutral":
                            detail["neutral_count"]
                    })

                aspect_table = pd.DataFrame(
                    aspect_rows
                )

                st.markdown(
                    aspect_table.to_html(
                        index=False
                    ),
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # SENTENCE DETAILS
                # ------------------------------------------------

                st.subheader(
                    "📌 Aspect Evidence"
                )

                for detail in aspect_details:

                    st.markdown(
                        f"**{detail['aspect']} → "
                        f"{detail['sentiment']}**"
                    )

                    for sentence in detail[
                        "sentences"
                    ]:

                        st.write(
                            "• " + sentence
                        )

            else:

                st.info(
                    "No specific aspect was detected."
                )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 AI Interpretation"
            )

            if result["sentiment"] == "Mixed":

                st.write(
                    "The feedback contains both "
                    "positive and negative opinions. "
                    "The system therefore classified "
                    "the overall sentiment as "
                    "**Mixed**."
                )

            else:

                st.write(
                    "The model identified the dominant "
                    f"sentiment as **{result['sentiment']}**, "
                    f"the dominant emotion as "
                    f"**{result['emotion']}**, and the "
                    f"primary aspect as **{result['aspect']}**."
                )

            st.caption(
                "Analysis method: "
                + result["method"]
            )


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif selected_page == "📊 Batch Analysis":

    st.header(
        "📊 Batch Feedback Analysis"
    )

    st.write(
        "Upload a CSV file containing a column "
        "named **feedback**."
    )

    st.info(
        "The system will analyze every feedback entry "
        "and add AI prediction columns."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )

            st.stop()

        if "feedback" not in df.columns:

            st.error(
                "Your CSV must contain a column "
                "named 'feedback'."
            )

        else:

            st.success(
                f"{len(df)} feedback entries found."
            )

            st.subheader(
                "📄 Dataset Preview"
            )

            preview_df = df.head(10)

            st.markdown(
                preview_df.to_html(
                    index=False
                ),
                unsafe_allow_html=True
            )

            st.divider()

            analyze_batch = st.button(
                "🚀 Analyze All Feedback",
                type="primary",
                use_container_width=True
            )

            if analyze_batch:

                results = []

                progress_bar = st.progress(
                    0.0
                )

                status_text = st.empty()

                total = len(df)

                for index, text_value in enumerate(
                    df["feedback"].astype(str)
                ):

                    status_text.write(
                        f"Analyzing "
                        f"{index + 1} of {total}..."
                    )

                    prediction = analyze_feedback(
                        text_value
                    )

                    detected_aspects = ", ".join(

                        detail["aspect"]

                        for detail in prediction[
                            "aspect_details"
                        ]
                    )

                    results.append({

                        "predicted_sentiment":
                            prediction["sentiment"],

                        "predicted_sentiment_confidence":
                            prediction[
                                "sentiment_confidence"
                            ],

                        "predicted_emotion":
                            prediction["emotion"],

                        "predicted_emotion_confidence":
                            prediction[
                                "emotion_confidence"
                            ],

                        "predicted_primary_aspect":
                            prediction["aspect"],

                        "predicted_aspect_confidence":
                            prediction[
                                "aspect_confidence"
                            ],

                        "detected_aspects":
                            detected_aspects
                    })

                    progress_bar.progress(
                        (index + 1) / total
                    )

                status_text.success(
                    "All feedback analyzed successfully."
                )

                result_df = pd.DataFrame(
                    results
                )

                final_df = pd.concat(
                    [
                        df.reset_index(
                            drop=True
                        ),
                        result_df
                    ],
                    axis=1
                )

                st.session_state[
                    "batch_results"
                ] = final_df

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            if "batch_results" in st.session_state:

                final_df = st.session_state[
                    "batch_results"
                ]

                st.divider()

                st.subheader(
                    "📋 Analysis Results"
                )

                result_preview = final_df.head(
                    100
                )

                st.markdown(
                    result_preview.to_html(
                        index=False
                    ),
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Showing first "
                    f"{min(100, len(final_df))} "
                    f"of {len(final_df)} records."
                )

                # Download full results

                csv_data = (
                    final_df
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    label=
                        "⬇️ Download Full Results",

                    data=
                        csv_data,

                    file_name=
                        "student_feedback_analysis.csv",

                    mime=
                        "text/csv",

                    use_container_width=True
                )


# ============================================================
# DASHBOARD
# ============================================================

elif selected_page == "📈 Dashboard":

    st.header(
        "📈 Feedback Analytics Dashboard"
    )

    if "batch_results" not in st.session_state:

        st.info(
            "Go to **📊 Batch Analysis**, upload your "
            "CSV, and click **Analyze All Feedback**."
        )

    else:

        df = st.session_state[
            "batch_results"
        ].copy()

        sentiment_column = (
            "predicted_sentiment"
        )

        emotion_column = (
            "predicted_emotion"
        )

        aspect_column = (
            "predicted_primary_aspect"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        total_feedback = int(
            len(df)
        )

        positive_count = int(
            df[sentiment_column]
            .eq("Positive")
            .sum()
        )

        negative_count = int(
            df[sentiment_column]
            .eq("Negative")
            .sum()
        )

        neutral_count = int(
            df[sentiment_column]
            .eq("Neutral")
            .sum()
        )

        mixed_count = int(
            df[sentiment_column]
            .eq("Mixed")
            .sum()
        )

        st.subheader(
            "📊 Overall Summary"
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Total",
            total_feedback
        )

        col2.metric(
            "Positive",
            positive_count
        )

        col3.metric(
            "Negative",
            negative_count
        )

        col4.metric(
            "Neutral",
            neutral_count
        )

        col5.metric(
            "Mixed",
            mixed_count
        )

        st.divider()

        # ----------------------------------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "💭 Sentiment Distribution"
        )

        sentiment_counts = (
            df[
                sentiment_column
            ]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig_sentiment = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            color="Sentiment",
            title="Overall Sentiment"
        )

        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )

        # ----------------------------------------------------
        # EMOTION DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "❤️ Emotion Distribution"
        )

        emotion_counts = (
            df[
                emotion_column
            ]
            .value_counts()
            .reset_index()
        )

        emotion_counts.columns = [
            "Emotion",
            "Count"
        ]

        fig_emotion = px.bar(
            emotion_counts,
            x="Emotion",
            y="Count",
            color="Emotion",
            title="Detected Emotions"
        )

        st.plotly_chart(
            fig_emotion,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PRIMARY ASPECT DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "🏫 Feedback by Primary Aspect"
        )

        aspect_counts = (
            df[
                aspect_column
            ]
            .value_counts()
            .reset_index()
        )

        aspect_counts.columns = [
            "Aspect",
            "Count"
        ]

        fig_aspect = px.bar(
            aspect_counts,
            x="Aspect",
            y="Count",
            color="Count",
            title="Main College Feedback Areas"
        )

        st.plotly_chart(
            fig_aspect,
            use_container_width=True
        )

        # ----------------------------------------------------
        # ASPECT VS SENTIMENT
        # ----------------------------------------------------

        st.subheader(
            "🔎 Aspect vs Sentiment"
        )

        cross_table = pd.crosstab(
            df[aspect_column],
            df[sentiment_column]
        )

        st.markdown(
            cross_table.to_html(),
            unsafe_allow_html=True
        )

        fig_heatmap = px.imshow(
            cross_table,
            text_auto=True,
            aspect="auto",
            title="Aspect-Sentiment Relationship",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )

        # ----------------------------------------------------
        # NEGATIVE AREAS
        # ----------------------------------------------------

        st.subheader(
            "⚠️ Negative Feedback by Area"
        )

        negative_df = df[
            df[sentiment_column]
            == "Negative"
        ]

        if len(negative_df) > 0:

            negative_counts = (
                negative_df[
                    aspect_column
                ]
                .value_counts()
                .reset_index()
            )

            negative_counts.columns = [
                "Aspect",
                "Negative Feedback"
            ]

            fig_negative = px.bar(
                negative_counts,
                x="Aspect",
                y="Negative Feedback",
                color="Negative Feedback",
                title="Areas Receiving Negative Feedback"
            )

            st.plotly_chart(
                fig_negative,
                use_container_width=True
            )

        else:

            st.info(
                "No negative feedback was predicted."
            )

        # ----------------------------------------------------
        # MIXED FEEDBACK
        # ----------------------------------------------------

        if mixed_count > 0:

            st.subheader(
                "🔀 Mixed Feedback"
            )

            mixed_df = df[
                df[sentiment_column]
                == "Mixed"
            ]

            st.markdown(
                mixed_df.head(20).to_html(
                    index=False
                ),
                unsafe_allow_html=True
            )


# ============================================================
# ABOUT
# ============================================================

elif selected_page == "ℹ️ About":

    st.header(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## 🎓 AI-Based Student Feedback
        ## Sentiment & Emotion Analyzer

        This application combines a trained
        **CNN + Bidirectional LSTM** model with
        NLP-based aspect detection.

        ### 🔬 Architecture

        ```text
        Student Feedback
                ↓
        Text Preprocessing
                ↓
        Tokenization
                ↓
        Word Embedding
                ↓
               CNN
                ↓
        Bidirectional LSTM
                ↓
         Shared Features
                ↓
          Sentiment
          Emotion
          Aspect
                ↓
          NLP Rule Layer
                ↓
        Multiple Aspect Analysis
                ↓
        Interactive Dashboard
        ```

        ### 💭 Sentiment

        - Positive
        - Negative
        - Neutral
        - Mixed

        ### ❤️ Emotion

        - Happy
        - Satisfied
        - Frustrated
        - Angry
        - Anxious
        - Disappointed
        - Neutral

        ### 🏫 Aspects

        - Teaching
        - Faculty
        - Courses
        - Laboratory
        - Library
        - Hostel
        - Transport
        - Infrastructure
        - Placement
        - Examination
        - Scholarship
        - Cafeteria

        ### 🛠 Technologies

        - Python
        - TensorFlow
        - Keras
        - NumPy
        - Pandas
        - Scikit-learn
        - Streamlit
        - Plotly

        ### 🎯 Objective

        Automatically transform unstructured student
        feedback into useful information about
        sentiment, emotion and college-related areas.

        ### 🔐 Privacy

        Real student feedback should be collected and
        processed according to appropriate institutional
        privacy and consent requirements.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🎓 Student Feedback AI

    <br>

    CNN + Bidirectional LSTM + NLP Rules

    <br>

    Deep Learning Mini Project

    </div>
    """,
    unsafe_allow_html=True
)