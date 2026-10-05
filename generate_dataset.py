import os
import random
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# SETTINGS
# ============================================================

random.seed(42)

os.makedirs("data", exist_ok=True)


# ============================================================
# STUDENT FEEDBACK PHRASES
# ============================================================

feedback_data = {

    "Teaching": {

        "Positive": [
            "the faculty explains concepts clearly",
            "the professor explains difficult topics well",
            "the teaching methods are effective",
            "the lecturer gives useful examples",
            "the classes are easy to understand",
            "the professor encourages students to ask questions",
            "the faculty uses practical examples during lectures",
            "the teaching quality is excellent",
            "the lectures are interesting and informative",
            "the faculty makes difficult subjects easier to understand"
        ],

        "Negative": [
            "the faculty does not explain difficult topics clearly",
            "the professor rushes through important concepts",
            "the teaching methods are confusing",
            "the lecturer does not give enough examples",
            "the classes are difficult to follow",
            "the professor does not answer student questions properly",
            "the faculty needs to improve the teaching methods",
            "the lectures are not interesting",
            "the professor spends too little time explaining concepts",
            "the teaching quality is poor"
        ],

        "Neutral": [
            "the faculty follows the planned syllabus",
            "the professor covers the required topics",
            "the teaching is average",
            "the lectures follow the timetable",
            "the teaching quality is acceptable",
            "the professor completes the scheduled topics",
            "the classes are conducted regularly",
            "the teaching follows the department plan",
            "the lectures are conducted as scheduled",
            "the current teaching system is normal"
        ]
    },


    "Faculty": {

        "Positive": [
            "the faculty members are supportive",
            "the professors are approachable",
            "teachers provide useful academic guidance",
            "faculty members respond quickly to doubts",
            "the professors are friendly with students",
            "teachers are willing to help outside class",
            "faculty members provide good guidance",
            "the professors listen to student concerns",
            "the faculty is cooperative",
            "teachers give helpful suggestions"
        ],

        "Negative": [
            "the faculty members are not supportive",
            "some professors are difficult to approach",
            "teachers do not provide enough guidance",
            "faculty members rarely respond to doubts",
            "some professors are not helpful",
            "teachers do not listen to student concerns",
            "faculty support is poor",
            "some teachers are not cooperative",
            "professors do not give enough academic guidance",
            "faculty members take too long to respond"
        ],

        "Neutral": [
            "the faculty members are available during office hours",
            "faculty interaction is normal",
            "the professors follow the department schedule",
            "faculty support is average",
            "teachers provide the required information",
            "faculty members are available when required",
            "professor interaction is acceptable",
            "teachers follow the normal academic process",
            "faculty communication is adequate",
            "the faculty system is normal"
        ]
    },


    "Courses": {

        "Positive": [
            "the course content is useful",
            "the subjects are interesting",
            "the syllabus is relevant to current technology",
            "the course material is informative",
            "the curriculum is well designed",
            "the subjects improve our knowledge",
            "the course structure is easy to follow",
            "the syllabus provides useful practical knowledge",
            "the course content is relevant to industry",
            "the subjects help students develop useful skills"
        ],

        "Negative": [
            "the course workload is too high",
            "the syllabus is difficult to complete",
            "some subjects contain outdated content",
            "the course material is confusing",
            "the curriculum needs improvement",
            "the syllabus contains too many topics",
            "the course structure is difficult to follow",
            "the subjects have too much theoretical content",
            "the course material is not updated regularly",
            "the workload creates unnecessary pressure"
        ],

        "Neutral": [
            "the course workload is manageable",
            "the syllabus covers the required topics",
            "the course content is average",
            "the subjects are reasonably structured",
            "the curriculum is acceptable",
            "the course follows the planned syllabus",
            "the subject content is adequate",
            "the course material is available",
            "the curriculum follows the university requirements",
            "the course structure is normal"
        ]
    },


    "Laboratory": {

        "Positive": [
            "the laboratory equipment is modern",
            "the practical sessions are useful",
            "the laboratory computers work well",
            "the lab facilities are excellent",
            "the practical classes improve our skills",
            "the laboratory environment is good",
            "the equipment is sufficient for practical work",
            "the lab sessions are well organized",
            "the practical exercises are useful",
            "the laboratory provides good hands-on experience"
        ],

        "Negative": [
            "the laboratory computers are very slow",
            "the lab equipment is outdated",
            "the laboratory facilities need improvement",
            "many computers do not work properly",
            "the practical sessions are poorly organized",
            "the laboratory has insufficient equipment",
            "the computers frequently stop working",
            "the lab facilities are not maintained properly",
            "the practical classes are difficult because of equipment problems",
            "the laboratory needs better computers"
        ],

        "Neutral": [
            "the laboratory facilities are acceptable",
            "the practical sessions follow the timetable",
            "the laboratory equipment is average",
            "the lab computers work normally",
            "the laboratory is available for practical sessions",
            "the equipment is sufficient for basic work",
            "the practical classes follow the schedule",
            "the lab environment is normal",
            "the laboratory facilities are adequate",
            "the current lab arrangements are acceptable"
        ]
    },


    "Library": {

        "Positive": [
            "the library has a good collection of books",
            "the library is quiet and comfortable",
            "the digital library is very useful",
            "the library facilities are excellent",
            "there are enough study spaces",
            "the library has many useful reference books",
            "the reading environment is comfortable",
            "the library provides useful online resources",
            "the library collection is helpful for students",
            "the study area is clean and peaceful"
        ],

        "Negative": [
            "the library does not have enough books",
            "the library closes too early",
            "there are not enough study spaces",
            "the library facilities need improvement",
            "many required books are unavailable",
            "the library collection is outdated",
            "there are too few seats for students",
            "the digital library resources are limited",
            "the library environment is uncomfortable",
            "important reference books are missing"
        ],

        "Neutral": [
            "the library has the required basic books",
            "the library facilities are average",
            "the library follows normal working hours",
            "the study space is acceptable",
            "the library collection is reasonable",
            "the required books are generally available",
            "the library follows the regular schedule",
            "the reading area is adequate",
            "the digital library is available",
            "the library facilities are normal"
        ]
    },


    "Hostel": {

        "Positive": [
            "the hostel rooms are clean",
            "the hostel facilities are good",
            "the hostel is well maintained",
            "the hostel environment is comfortable",
            "the hostel staff are helpful",
            "the rooms are spacious and clean",
            "the hostel provides useful facilities",
            "the maintenance service is good",
            "the hostel environment is peaceful",
            "the accommodation is comfortable"
        ],

        "Negative": [
            "the hostel rooms are dirty",
            "the hostel facilities need improvement",
            "the hostel food is poor",
            "the hostel is not properly maintained",
            "the hostel environment is uncomfortable",
            "the rooms require better maintenance",
            "the hostel facilities are insufficient",
            "the food quality is not good",
            "maintenance complaints are not handled quickly",
            "the hostel environment is noisy"
        ],

        "Neutral": [
            "the hostel facilities are average",
            "the hostel rooms are acceptable",
            "the hostel follows the required rules",
            "the hostel environment is normal",
            "the hostel services are satisfactory",
            "the rooms meet basic requirements",
            "the hostel follows the regular schedule",
            "the maintenance service is adequate",
            "the accommodation is acceptable",
            "the hostel facilities are normal"
        ]
    },


    "Transport": {

        "Positive": [
            "the college bus service is convenient",
            "the buses are well maintained",
            "transport facilities are useful",
            "the bus routes are convenient",
            "the transport service is reliable",
            "the buses are usually on time",
            "the bus service makes travel easier",
            "the transport facilities are comfortable",
            "the routes are useful for students",
            "the college provides good transport support"
        ],

        "Negative": [
            "the college buses are overcrowded",
            "the bus service is unreliable",
            "the buses are often late",
            "transport facilities need improvement",
            "the bus routes are inconvenient",
            "the buses do not follow the schedule",
            "the transport service causes delays",
            "the buses are uncomfortable",
            "there are not enough buses",
            "the bus service is poorly managed"
        ],

        "Neutral": [
            "the transport service is normal",
            "the buses follow the regular schedule",
            "the transport facilities are acceptable",
            "the bus service is average",
            "the current routes are adequate",
            "the buses operate according to the timetable",
            "the transport system meets basic requirements",
            "the bus facilities are reasonable",
            "the current transport service is acceptable",
            "the college buses are available as scheduled"
        ]
    },


    "Infrastructure": {

        "Positive": [
            "the classroom facilities are excellent",
            "the campus WiFi works very well",
            "the classrooms are comfortable",
            "the campus infrastructure is modern",
            "the college facilities are well maintained",
            "the campus has good learning facilities",
            "the classrooms have useful equipment",
            "the campus environment is comfortable",
            "the internet facilities are reliable",
            "the infrastructure supports student learning"
        ],

        "Negative": [
            "the WiFi connection is very poor",
            "the classrooms are uncomfortable",
            "the infrastructure needs improvement",
            "the campus facilities are poorly maintained",
            "the classrooms are too hot",
            "the internet connection frequently fails",
            "the classrooms need better equipment",
            "the campus facilities are outdated",
            "the infrastructure is not maintained properly",
            "the learning environment is uncomfortable"
        ],

        "Neutral": [
            "the campus infrastructure is average",
            "the WiFi works normally",
            "the classrooms are acceptable",
            "the facilities are reasonably maintained",
            "the infrastructure meets basic requirements",
            "the classroom facilities are adequate",
            "the internet service is acceptable",
            "the campus facilities are normal",
            "the infrastructure is sufficient for regular use",
            "the classrooms follow the standard facilities"
        ]
    },


    "Placement": {

        "Positive": [
            "the placement training is very useful",
            "the placement department provides good support",
            "many companies visit the campus",
            "the career guidance sessions are helpful",
            "the placement opportunities are good",
            "students receive useful interview training",
            "the placement team provides good guidance",
            "the companies offer relevant opportunities",
            "the training improves employability skills",
            "the placement process is well organized"
        ],

        "Negative": [
            "the placement opportunities are limited",
            "very few companies visit the campus",
            "placement training needs improvement",
            "career guidance is not sufficient",
            "the placement department provides little support",
            "students do not receive enough interview training",
            "the placement opportunities are not diverse",
            "the number of companies is too low",
            "the placement preparation is inadequate",
            "students need better career guidance"
        ],

        "Neutral": [
            "the placement support is average",
            "companies visit the campus regularly",
            "the placement training is acceptable",
            "career guidance is provided",
            "the placement process follows the normal schedule",
            "the placement department provides basic support",
            "the current placement opportunities are reasonable",
            "the training is available for students",
            "the placement process is normal",
            "the career support system is adequate"
        ]
    },


    "Examination": {

        "Positive": [
            "the examination process is well organized",
            "the exam timetable is clearly communicated",
            "the examination system is convenient",
            "the exams are conducted properly",
            "the evaluation process is transparent",
            "the examination schedule is easy to understand",
            "the exam arrangements are good",
            "the evaluation system is organized",
            "the examination rules are clearly communicated",
            "the exam process is convenient for students"
        ],

        "Negative": [
            "the examination schedule is confusing",
            "the exam timetable was announced late",
            "the examination process is stressful",
            "the evaluation process needs improvement",
            "the exam arrangements are poor",
            "the timetable is difficult to understand",
            "students receive exam information too late",
            "the examination process creates unnecessary stress",
            "the evaluation system is unclear",
            "the exam arrangements are not convenient"
        ],

        "Neutral": [
            "the examination process is normal",
            "the exams follow the academic schedule",
            "the examination system is acceptable",
            "the timetable is available on time",
            "the evaluation process follows the rules",
            "the examination schedule is standard",
            "the exams are conducted according to the plan",
            "the evaluation system is adequate",
            "the examination procedure is normal",
            "the current exam system is acceptable"
        ]
    },


    "Scholarship": {

        "Positive": [
            "the scholarship information is clearly provided",
            "the scholarship process is helpful",
            "students receive useful scholarship guidance",
            "the scholarship application process is simple",
            "the scholarship information is easy to understand",
            "the application instructions are clear",
            "the scholarship support is useful",
            "students receive timely information",
            "the scholarship procedure is convenient",
            "the college provides good scholarship guidance"
        ],

        "Negative": [
            "the scholarship process is complicated",
            "scholarship information is difficult to find",
            "the application process takes too long",
            "students do not receive enough guidance",
            "the scholarship procedure needs improvement",
            "the application requirements are confusing",
            "scholarship updates are not communicated properly",
            "students face delays in the scholarship process",
            "the scholarship procedure is difficult",
            "the available guidance is insufficient"
        ],

        "Neutral": [
            "the scholarship information is available",
            "the application process is normal",
            "scholarship details are provided",
            "the scholarship procedure is acceptable",
            "students can access the required information",
            "the scholarship application follows the normal process",
            "the required documents are listed",
            "the scholarship information is adequate",
            "the current procedure is acceptable",
            "students receive basic scholarship information"
        ]
    },


    "Cafeteria": {

        "Positive": [
            "the cafeteria food is good",
            "the cafeteria is clean",
            "the food quality is satisfactory",
            "the cafeteria provides good variety",
            "the cafeteria service is fast",
            "the food is fresh and tasty",
            "the cafeteria environment is comfortable",
            "the staff provide good service",
            "the cafeteria has reasonable options",
            "the food quality is consistently good"
        ],

        "Negative": [
            "the cafeteria food is poor",
            "the food is too expensive",
            "the cafeteria is not clean",
            "the food quality needs improvement",
            "the cafeteria service is slow",
            "the food is not fresh",
            "the cafeteria has limited variety",
            "the cafeteria environment is uncomfortable",
            "the food prices are too high",
            "the service is poorly managed"
        ],

        "Neutral": [
            "the cafeteria food is average",
            "the cafeteria prices are reasonable",
            "the cafeteria is normally maintained",
            "the food variety is acceptable",
            "the cafeteria service is normal",
            "the food quality is adequate",
            "the cafeteria follows the regular schedule",
            "the available options are reasonable",
            "the cafeteria environment is acceptable",
            "the cafeteria facilities are normal"
        ]
    }
}


# ============================================================
# EMOTION CLASSES
# ============================================================

emotion_map = {

    "Positive": [
        "Happy",
        "Satisfied"
    ],

    "Negative": [
        "Frustrated",
        "Angry",
        "Anxious",
        "Disappointed"
    ],

    "Neutral": [
        "Neutral"
    ]
}


# ============================================================
# SENTENCE PATTERNS
# ============================================================

positive_patterns = [

    "Overall, {}.",
    "I am happy that {}.",
    "I am satisfied because {}.",
    "I really appreciate that {}.",
    "I am pleased that {}.",
    "From my experience, {}.",
    "I have had a good experience because {}.",
    "I feel positive about the fact that {}.",
    "As a student, I am happy that {}.",
    "My experience has been good because {}."
]


negative_patterns = [

    "Overall, {}.",
    "I am frustrated because {}.",
    "I am unhappy because {}.",
    "I am disappointed because {}.",
    "I am worried because {}.",
    "I feel that {} is a serious problem.",
    "As a student, I am concerned that {}.",
    "My experience has been difficult because {}.",
    "I am not satisfied because {}.",
    "I strongly feel that {}."
]


neutral_patterns = [

    "Overall, {}.",
    "In general, {}.",
    "From my experience, {}.",
    "Currently, {}.",
    "I observed that {}.",
    "For me, {}.",
    "At present, {}.",
    "In my opinion, {}.",
    "The current situation is that {}.",
    "Generally, {}."
]


# ============================================================
# EMOTION-SPECIFIC PATTERNS
# ============================================================

emotion_patterns = {

    "Happy": [
        "I am happy that {}.",
        "I really enjoy that {}.",
        "I feel happy because {}."
    ],

    "Satisfied": [
        "I am satisfied because {}.",
        "I am pleased that {}.",
        "I am happy with the fact that {}."
    ],

    "Frustrated": [
        "I am frustrated because {}.",
        "I feel frustrated because {}.",
        "This frustrates me because {}."
    ],

    "Angry": [
        "I am angry because {}.",
        "I am very unhappy that {}.",
        "This makes me angry because {}."
    ],

    "Anxious": [
        "I feel worried because {}.",
        "I am concerned because {}.",
        "This makes me anxious because {}."
    ],

    "Disappointed": [
        "I am disappointed because {}.",
        "I feel disappointed that {}.",
        "This has been disappointing because {}."
    ],

    "Neutral": [
        "I observed that {}.",
        "From my experience, {}.",
        "Currently, {}."
    ]
}


# ============================================================
# OPTIONAL ENDINGS
# ============================================================

endings = {

    "Positive": [
        "",
        " Overall, I am happy with the situation.",
        " This has improved my college experience.",
        " I appreciate the effort taken by the college.",
        " This is helpful for students."
    ],

    "Negative": [
        "",
        " This should be improved soon.",
        " I hope the college addresses this issue.",
        " This is affecting the student experience.",
        " Students need better support in this area."
    ],

    "Neutral": [
        "",
        " This is acceptable for now.",
        " This is the current situation.",
        " I do not have a strong opinion about it.",
        " The current arrangement is reasonable."
    ]
}


# ============================================================
# GENERATE DATASET
# ============================================================

records = []

start_date = datetime(2026, 1, 1)

record_id = 1


for aspect, sentiment_groups in feedback_data.items():

    for sentiment, phrases in sentiment_groups.items():

        generated = set()

        # 50 samples per sentiment for each aspect
        while len(generated) < 50:

            phrase = random.choice(
                phrases
            )

            emotion = random.choice(
                emotion_map[sentiment]
            )

            # Use emotion-specific wording
            # for most examples.
            if random.random() < 0.65:

                pattern = random.choice(
                    emotion_patterns[emotion]
                )

                sentence = pattern.format(
                    phrase
                )

            else:

                if sentiment == "Positive":

                    pattern = random.choice(
                        positive_patterns
                    )

                elif sentiment == "Negative":

                    pattern = random.choice(
                        negative_patterns
                    )

                else:

                    pattern = random.choice(
                        neutral_patterns
                    )

                sentence = pattern.format(
                    phrase
                )

            # Add an optional ending
            sentence = (
                sentence
                + random.choice(
                    endings[sentiment]
                )
            )

            sentence = sentence.strip()

            if sentence in generated:
                continue

            generated.add(
                sentence
            )

            random_date = (
                start_date
                + timedelta(
                    days=random.randint(
                        0,
                        270
                    )
                )
            )

            department = random.choice([
                "AIDS",
                "CSE",
                "ECE",
                "EEE",
                "IT",
                "MECH",
                "CIVIL"
            ])

            semester = random.randint(
                1,
                8
            )

            records.append({

                "id": record_id,

                "date":
                    random_date.strftime(
                        "%Y-%m-%d"
                    ),

                "department":
                    department,

                "semester":
                    semester,

                "feedback":
                    sentence,

                "sentiment":
                    sentiment,

                "emotion":
                    emotion,

                "aspect":
                    aspect
            })

            record_id += 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


# Shuffle the dataset

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/feedback_data.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("IMPROVED STUDENT FEEDBACK DATASET")
print("=" * 70)

print()

print(
    "Total records:",
    len(df)
)

print()

print(
    "Sentiment distribution:"
)

print(
    df["sentiment"].value_counts()
)

print()

print(
    "Emotion distribution:"
)

print(
    df["emotion"].value_counts()
)

print()

print(
    "Aspect distribution:"
)

print(
    df["aspect"].value_counts()
)

print()

print(
    "Dataset saved to:"
)

print(
    output_file
)

print()

print("=" * 70)
print("DATASET GENERATION COMPLETE")
print("=" * 70)