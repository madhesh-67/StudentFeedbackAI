import pickle
import re

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "model/feedback_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
ENCODER_PATH = "model/label_encoders.pkl"

MAX_SEQUENCE_LENGTH = 50


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    tokenizer = pickle.load(file)


# ============================================================
# LOAD LABEL ENCODERS
# ============================================================

with open(
    ENCODER_PATH,
    "rb"
) as file:

    encoders = pickle.load(file)


# ============================================================
# ASPECT KEYWORDS
# ============================================================

ASPECT_KEYWORDS = {

    "Teaching": [
        "teaching",
        "teach",
        "teaches",
        "taught",
        "lecture",
        "lectures",
        "explain",
        "explains",
        "explanation",
        "teaching method",
        "teaching methods"
    ],

    "Faculty": [
        "faculty",
        "professor",
        "professors",
        "teacher",
        "teachers",
        "lecturer",
        "faculty member",
        "faculty members"
    ],

    "Courses": [
        "course",
        "courses",
        "subject",
        "subjects",
        "syllabus",
        "curriculum",
        "coursework",
        "workload",
        "course content",
        "course material"
    ],

    "Laboratory": [
        "laboratory",
        "lab",
        "practical",
        "practicals",
        "experiment",
        "experiments",
        "lab computer",
        "lab computers",
        "laboratory computer",
        "laboratory computers"
    ],

    "Library": [
        "library",
        "book",
        "books",
        "reading area",
        "study space",
        "digital library",
        "reference book",
        "reference books"
    ],

    "Hostel": [
        "hostel",
        "hostel room",
        "hostel rooms",
        "accommodation",
        "hostel food",
        "hostel facility",
        "hostel facilities"
    ],

    "Transport": [
        "transport",
        "bus",
        "buses",
        "bus service",
        "bus route",
        "bus routes",
        "travel"
    ],

    "Infrastructure": [
        "infrastructure",
        "wifi",
        "wi-fi",
        "internet",
        "internet connection",
        "network",
        "classroom",
        "classrooms",
        "campus",
        "air conditioning",
        "air conditioner",
        "ac facility",
        "ac facilities",
        "college facilities"
    ],

    "Placement": [
        "placement",
        "placements",
        "company",
        "companies",
        "career",
        "career guidance",
        "interview",
        "interviews",
        "employability"
    ],

    "Examination": [
        "exam",
        "exams",
        "examination",
        "examinations",
        "exam timetable",
        "timetable",
        "evaluation",
        "evaluation process",
        "assessment"
    ],

    "Scholarship": [
        "scholarship",
        "scholarships",
        "scholarship application",
        "scholarship process",
        "application process",
        "financial aid"
    ],

    "Cafeteria": [
        "cafeteria",
        "canteen",
        "dining",
        "cafeteria food",
        "canteen food",
        "food quality",
        "food variety"
    ]
}


# ============================================================
# POSITIVE WORDS
# ============================================================

POSITIVE_WORDS = [

    "good",
    "great",
    "excellent",
    "amazing",
    "useful",
    "helpful",
    "supportive",
    "comfortable",
    "clean",
    "modern",
    "reliable",
    "convenient",
    "interesting",
    "effective",
    "clear",
    "clearly",
    "friendly",
    "approachable",
    "satisfied",
    "happy",
    "pleased",
    "enjoy",
    "enjoyed",
    "best",
    "positive",
    "fresh",
    "tasty",
    "fast",
    "well maintained",
    "well organized"
]


# ============================================================
# NEGATIVE WORDS
# ============================================================

NEGATIVE_WORDS = [

    "bad",
    "poor",
    "terrible",
    "worst",
    "slow",
    "difficult",
    "confusing",
    "outdated",
    "dirty",
    "uncomfortable",
    "unreliable",
    "late",
    "crowded",
    "overcrowded",
    "expensive",
    "limited",
    "stressful",
    "stress",
    "problem",
    "problems",
    "issue",
    "issues",
    "frustrated",
    "frustrating",
    "angry",
    "worried",
    "anxious",
    "disappointed",
    "poorly",
    "insufficient",
    "inadequate",
    "unavailable",
    "need improvement",
    "needs improvement",
    "not enough",
    "do not",
    "does not",
    "not supportive",
    "not helpful",
    "not good",
    "not clear",
    "not working",
    "not properly"
]


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s'-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SPLIT FEEDBACK INTO SENTENCES
# ============================================================

def split_sentences(text):

    text = str(text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# ============================================================
# DETECT ASPECTS
# ============================================================

def detect_aspects(text):

    text_lower = text.lower()

    detected = []

    for aspect, keywords in ASPECT_KEYWORDS.items():

        for keyword in keywords:

            if keyword.lower() in text_lower:

                detected.append(
                    aspect
                )

                break

    return detected


# ============================================================
# RULE-BASED SENTIMENT
# ============================================================

def rule_based_sentiment(
    text,
    neural_sentiment
):

    text_lower = text.lower()

    positive_score = 0
    negative_score = 0


    # --------------------------------------------------------
    # Count positive words
    # --------------------------------------------------------

    for word in POSITIVE_WORDS:

        if word in text_lower:

            positive_score += 1


    # --------------------------------------------------------
    # Count negative words
    # --------------------------------------------------------

    for word in NEGATIVE_WORDS:

        if word in text_lower:

            negative_score += 1


    # --------------------------------------------------------
    # Strong negative phrases
    # --------------------------------------------------------

    strong_negative_phrases = [

        "very slow",
        "very poor",
        "extremely slow",
        "extremely poor",
        "not working",
        "does not work",
        "do not work",
        "needs to improve",
        "need to improve",
        "not properly"
    ]


    for phrase in strong_negative_phrases:

        if phrase in text_lower:

            negative_score += 2


    # --------------------------------------------------------
    # Determine sentiment
    # --------------------------------------------------------

    if (
        positive_score > 0
        and negative_score > 0
    ):

        return "Mixed"


    if negative_score > positive_score:

        return "Negative"


    if positive_score > negative_score:

        return "Positive"


    return neural_sentiment


# ============================================================
# NEURAL NETWORK PREDICTION
# ============================================================

def neural_prediction(
    sentence
):

    cleaned = clean_text(
        sentence
    )


    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )


    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )


    predictions = model.predict(
        padded,
        verbose=0
    )


    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    sentiment_probs = predictions[0][0]

    sentiment_index = int(
        np.argmax(
            sentiment_probs
        )
    )


    sentiment = encoders[
        "sentiment"
    ].inverse_transform(
        [sentiment_index]
    )[0]


    sentiment_confidence = float(
        sentiment_probs[
            sentiment_index
        ]
    )


    # --------------------------------------------------------
    # EMOTION
    # --------------------------------------------------------

    emotion_probs = predictions[1][0]

    emotion_index = int(
        np.argmax(
            emotion_probs
        )
    )


    emotion = encoders[
        "emotion"
    ].inverse_transform(
        [emotion_index]
    )[0]


    emotion_confidence = float(
        emotion_probs[
            emotion_index
        ]
    )


    # --------------------------------------------------------
    # ASPECT
    # --------------------------------------------------------

    aspect_probs = predictions[2][0]

    aspect_index = int(
        np.argmax(
            aspect_probs
        )
    )


    aspect = encoders[
        "aspect"
    ].inverse_transform(
        [aspect_index]
    )[0]


    aspect_confidence = float(
        aspect_probs[
            aspect_index
        ]
    )


    # --------------------------------------------------------
    # ASPECT PROBABILITIES
    # --------------------------------------------------------

    aspect_probability_dict = {}

    aspect_classes = encoders[
        "aspect"
    ].classes_


    for class_name, probability in zip(
        aspect_classes,
        aspect_probs
    ):

        aspect_probability_dict[
            class_name
        ] = float(
            probability
        )


    return {

        "sentiment":
            sentiment,

        "sentiment_confidence":
            sentiment_confidence,

        "emotion":
            emotion,

        "emotion_confidence":
            emotion_confidence,

        "aspect":
            aspect,

        "aspect_confidence":
            aspect_confidence,

        "aspect_probabilities":
            aspect_probability_dict
    }


# ============================================================
# ANALYZE COMPLETE FEEDBACK
# ============================================================

def analyze_feedback(
    feedback
):

    feedback = str(
        feedback
    ).strip()


    if not feedback:

        return {

            "sentiment":
                "Unknown",

            "sentiment_confidence":
                0.0,

            "emotion":
                "Unknown",

            "emotion_confidence":
                0.0,

            "aspect":
                "Unknown",

            "aspect_confidence":
                0.0,

            "aspect_details":
                [],

            "method":
                "Hybrid CNN-BiLSTM + NLP Rules"
        }


    # --------------------------------------------------------
    # Split into individual sentences
    # --------------------------------------------------------

    sentences = split_sentences(
        feedback
    )


    sentence_results = []


    for sentence in sentences:

        neural = neural_prediction(
            sentence
        )


        # Rule-based sentiment
        final_sentiment = (
            rule_based_sentiment(
                sentence,
                neural["sentiment"]
            )
        )


        # Detect one or more aspects
        detected_aspects = detect_aspects(
            sentence
        )


        # If no keyword is found,
        # use the neural prediction.
        if not detected_aspects:

            detected_aspects = [
                neural["aspect"]
            ]


        sentence_results.append({

            "sentence":
                sentence,

            "sentiment":
                final_sentiment,

            "sentiment_confidence":
                neural[
                    "sentiment_confidence"
                ],

            "emotion":
                neural[
                    "emotion"
                ],

            "emotion_confidence":
                neural[
                    "emotion_confidence"
                ],

            "detected_aspects":
                detected_aspects,

            "aspect_probabilities":
                neural[
                    "aspect_probabilities"
                ]
        })


    # ========================================================
    # OVERALL SENTIMENT
    # ========================================================

    positive_present = False
    negative_present = False


    for item in sentence_results:

        if item["sentiment"] in [
            "Positive",
            "Mixed"
        ]:

            positive_present = True


        if item["sentiment"] in [
            "Negative",
            "Mixed"
        ]:

            negative_present = True


    if (
        positive_present
        and negative_present
    ):

        overall_sentiment = "Mixed"

    elif positive_present:

        overall_sentiment = "Positive"

    elif negative_present:

        overall_sentiment = "Negative"

    else:

        overall_sentiment = "Neutral"


    sentiment_confidences = [

        item[
            "sentiment_confidence"
        ]

        for item in sentence_results
    ]


    if sentiment_confidences:

        overall_sentiment_confidence = float(
            np.mean(
                sentiment_confidences
            )
        )

    else:

        overall_sentiment_confidence = 0.0


    # ========================================================
    # OVERALL EMOTION
    # ========================================================

    emotion_scores = {}


    for item in sentence_results:

        emotion = item[
            "emotion"
        ]

        confidence = item[
            "emotion_confidence"
        ]


        emotion_scores[
            emotion
        ] = (
            emotion_scores.get(
                emotion,
                0.0
            )
            + confidence
        )


    if emotion_scores:

        overall_emotion = max(
            emotion_scores,
            key=emotion_scores.get
        )


        emotion_confidences = [

            item[
                "emotion_confidence"
            ]

            for item in sentence_results

            if item["emotion"]
            == overall_emotion
        ]


        if emotion_confidences:

            overall_emotion_confidence = float(
                np.mean(
                    emotion_confidences
                )
            )

        else:

            overall_emotion_confidence = 0.0

    else:

        overall_emotion = "Neutral"

        overall_emotion_confidence = 0.0


    # ========================================================
    # GROUP ASPECTS
    # ========================================================

    aspect_groups = {}


    for item in sentence_results:

        for aspect in item[
            "detected_aspects"
        ]:

            if aspect not in aspect_groups:

                aspect_groups[
                    aspect
                ] = []


            model_probability = (
                item[
                    "aspect_probabilities"
                ].get(
                    aspect,
                    0.0
                )
            )


            aspect_groups[
                aspect
            ].append({

                "sentence":
                    item[
                        "sentence"
                    ],

                "sentiment":
                    item[
                        "sentiment"
                    ],

                "model_probability":
                    model_probability
            })


    # ========================================================
    # ASPECT DETAILS
    # ========================================================

    aspect_details = []


    for aspect, items in aspect_groups.items():

        # ----------------------------------------------------
        # IMPORTANT:
        # A Mixed sentence contributes to BOTH positive
        # and negative counts.
        # ----------------------------------------------------

        positive_count = sum(

            1

            for item in items

            if item["sentiment"]
            in [
                "Positive",
                "Mixed"
            ]
        )


        negative_count = sum(

            1

            for item in items

            if item["sentiment"]
            in [
                "Negative",
                "Mixed"
            ]
        )


        neutral_count = sum(

            1

            for item in items

            if item["sentiment"]
            == "Neutral"
        )


        mixed_count = sum(

            1

            for item in items

            if item["sentiment"]
            == "Mixed"
        )


        # ----------------------------------------------------
        # Determine aspect sentiment
        # ----------------------------------------------------

        if mixed_count > 0:

            aspect_sentiment = "Mixed"

        elif (
            positive_count > 0
            and negative_count > 0
        ):

            aspect_sentiment = "Mixed"

        elif positive_count > negative_count:

            aspect_sentiment = "Positive"

        elif negative_count > positive_count:

            aspect_sentiment = "Negative"

        else:

            aspect_sentiment = "Neutral"


        model_probabilities = [

            item[
                "model_probability"
            ]

            for item in items
        ]


        if model_probabilities:

            aspect_confidence = float(
                np.mean(
                    model_probabilities
                )
            )

        else:

            aspect_confidence = 0.0


        aspect_details.append({

            "aspect":
                aspect,

            "sentiment":
                aspect_sentiment,

            "confidence":
                aspect_confidence,

            "positive_count":
                positive_count,

            "negative_count":
                negative_count,

            "neutral_count":
                neutral_count,

            "sentences":
                [
                    item[
                        "sentence"
                    ]

                    for item in items
                ]
        })


    # ========================================================
    # PRIMARY ASPECT
    # ========================================================

    if aspect_details:

        primary = max(

            aspect_details,

            key=lambda item: (
                len(
                    item[
                        "sentences"
                    ]
                ),

                item[
                    "confidence"
                ]
            )
        )


        primary_aspect = primary[
            "aspect"
        ]


        primary_aspect_confidence = float(
            primary[
                "confidence"
            ]
        )

    else:

        primary_aspect = "Unknown"

        primary_aspect_confidence = 0.0


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "sentiment":
            overall_sentiment,

        "sentiment_confidence":
            overall_sentiment_confidence,

        "emotion":
            overall_emotion,

        "emotion_confidence":
            overall_emotion_confidence,

        "aspect":
            primary_aspect,

        "aspect_confidence":
            primary_aspect_confidence,

        "aspect_details":
            aspect_details,

        "method":
            "Hybrid CNN-BiLSTM + NLP Rules"
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("STUDENT FEEDBACK AI ANALYZER")
    print("=" * 70)


    feedback = input(
        "\nEnter student feedback: "
    )


    result = analyze_feedback(
        feedback
    )


    print()
    print("=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)


    print()
    print(
        "Sentiment :",
        result["sentiment"]
    )


    print(
        "Confidence:",
        f"{result['sentiment_confidence']:.2%}"
    )


    print()
    print(
        "Emotion   :",
        result["emotion"]
    )


    print(
        "Confidence:",
        f"{result['emotion_confidence']:.2%}"
    )


    print()
    print(
        "Primary Aspect:",
        result["aspect"]
    )


    print(
        "Confidence:",
        f"{result['aspect_confidence']:.2%}"
    )


    print()
    print(
        "Analysis Method:",
        result["method"]
    )


    print()
    print("=" * 70)
    print("ASPECT-LEVEL ANALYSIS")
    print("=" * 70)


    if not result[
        "aspect_details"
    ]:

        print(
            "\nNo specific aspect detected."
        )

    else:

        for detail in result[
            "aspect_details"
        ]:

            print()

            print(
                "Aspect:",
                detail["aspect"]
            )

            print(
                "Sentiment:",
                detail["sentiment"]
            )

            print(
                "Model support:",
                f"{detail['confidence']:.2%}"
            )

            print(
                "Positive:",
                detail["positive_count"]
            )

            print(
                "Negative:",
                detail["negative_count"]
            )

            print(
                "Neutral:",
                detail["neutral_count"]
            )

            print(
                "Sentences:"
            )


            for sentence in detail[
                "sentences"
            ]:

                print(
                    " -",
                    sentence
                )


    print()
    print("=" * 70)