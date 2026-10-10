from flask import Flask, render_template, request
import pandas as pd
import joblib
import json


# ---------------------------------------------------------
# FLASK APPLICATION
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# LOAD DATASET AND TRAINED MODEL
# ---------------------------------------------------------

data = pd.read_csv("beer-servings-cleaned.csv")

model = joblib.load("best_model.pkl")


# ---------------------------------------------------------
# PREPARE DROPDOWN VALUES
# ---------------------------------------------------------

countries = sorted(
    data["country"]
    .dropna()
    .unique()
    .tolist()
)

continents = sorted(
    data["continent"]
    .dropna()
    .unique()
    .tolist()
)


# ---------------------------------------------------------
# COUNTRY -> CONTINENT MAPPING
# ---------------------------------------------------------

country_continent_map = (
    data[["country", "continent"]]
    .dropna()
    .drop_duplicates(subset="country")
    .set_index("country")["continent"]
    .to_dict()
)


# ---------------------------------------------------------
# DASHBOARD KPI VALUES
# ---------------------------------------------------------

total_countries = int(
    data["country"].nunique()
)

avg_beer = round(
    data["beer_servings"].mean(),
    1
)

avg_spirit = round(
    data["spirit_servings"].mean(),
    1
)

avg_wine = round(
    data["wine_servings"].mean(),
    1
)

avg_alcohol = round(
    data["total_litres_of_pure_alcohol"].mean(),
    2
)


# ---------------------------------------------------------
# CONTINENT ANALYSIS
# ---------------------------------------------------------

continent_analysis = (
    data
    .groupby("continent")[
        "total_litres_of_pure_alcohol"
    ]
    .mean()
    .reset_index()
    .sort_values(
        "total_litres_of_pure_alcohol",
        ascending=False
    )
)

continent_labels = (
    continent_analysis["continent"]
    .tolist()
)

continent_values = (
    continent_analysis[
        "total_litres_of_pure_alcohol"
    ]
    .round(2)
    .tolist()
)


# ---------------------------------------------------------
# TOP 10 COUNTRIES
# ---------------------------------------------------------

top_countries = (
    data
    .nlargest(
        10,
        "total_litres_of_pure_alcohol"
    )[
        [
            "country",
            "total_litres_of_pure_alcohol"
        ]
    ]
    .sort_values(
        "total_litres_of_pure_alcohol",
        ascending=True
    )
)

top_country_labels = (
    top_countries["country"]
    .tolist()
)

top_country_values = (
    top_countries[
        "total_litres_of_pure_alcohol"
    ]
    .round(2)
    .tolist()
)


# ---------------------------------------------------------
# BEVERAGE AVERAGES
# ---------------------------------------------------------

beverage_labels = [
    "Beer",
    "Spirit",
    "Wine"
]

beverage_values = [
    round(
        data["beer_servings"].mean(),
        2
    ),
    round(
        data["spirit_servings"].mean(),
        2
    ),
    round(
        data["wine_servings"].mean(),
        2
    )
]


# ---------------------------------------------------------
# DASHBOARD ROUTE
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",

        total_countries=total_countries,
        avg_beer=avg_beer,
        avg_spirit=avg_spirit,
        avg_wine=avg_wine,
        avg_alcohol=avg_alcohol,

        continent_labels=json.dumps(
            continent_labels
        ),

        continent_values=json.dumps(
            continent_values
        ),

        top_country_labels=json.dumps(
            top_country_labels
        ),

        top_country_values=json.dumps(
            top_country_values
        ),

        beverage_labels=json.dumps(
            beverage_labels
        ),

        beverage_values=json.dumps(
            beverage_values
        )
    )


# ---------------------------------------------------------
# PREDICTION ROUTE
# ---------------------------------------------------------

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    prediction = None
    error = None

    selected_country = ""
    selected_continent = ""

    beer = ""
    spirit = ""
    wine = ""

    if request.method == "POST":

        try:

            # ---------------------------------------------
            # READ USER INPUT
            # ---------------------------------------------

            selected_country = (
                request.form["country"]
            )

            beer = float(
                request.form["beer_servings"]
            )

            spirit = float(
                request.form["spirit_servings"]
            )

            wine = float(
                request.form["wine_servings"]
            )


            # ---------------------------------------------
            # VALIDATE COUNTRY
            # ---------------------------------------------

            if (
                selected_country
                not in country_continent_map
            ):

                error = (
                    "Please select a valid country."
                )


            # ---------------------------------------------
            # VALIDATE NUMERICAL VALUES
            # ---------------------------------------------

            elif (
                beer < 0
                or spirit < 0
                or wine < 0
            ):

                error = (
                    "Serving values cannot "
                    "be negative."
                )


            else:

                # -----------------------------------------
                # GET CORRECT CONTINENT FROM DATASET
                # -----------------------------------------

                selected_continent = (
                    country_continent_map[
                        selected_country
                    ]
                )


                # -----------------------------------------
                # CREATE MODEL INPUT
                # -----------------------------------------

                input_data = pd.DataFrame({

                    "country": [
                        selected_country
                    ],

                    "beer_servings": [
                        beer
                    ],

                    "spirit_servings": [
                        spirit
                    ],

                    "wine_servings": [
                        wine
                    ],

                    "continent": [
                        selected_continent
                    ]

                })


                # -----------------------------------------
                # MAKE PREDICTION
                # -----------------------------------------

                result = model.predict(
                    input_data
                )[0]


                # Prevent impossible negative consumption

                result = max(
                    0,
                    float(result)
                )


                prediction = round(
                    result,
                    2
                )


        except ValueError:

            error = (
                "Please enter valid "
                "numerical values."
            )


        except Exception as e:

            print(
                "Prediction Error:",
                e
            )

            error = (
                "Unable to generate the prediction. "
                "Please check the entered values."
            )


    # If GET request, or country hasn't yet been submitted

    if (
        selected_country
        and selected_country
        in country_continent_map
    ):

        selected_continent = (
            country_continent_map[
                selected_country
            ]
        )


    return render_template(
        "predict.html",

        countries=countries,
        continents=continents,

        country_continent_map=(
            country_continent_map
        ),

        prediction=prediction,
        error=error,

        selected_country=(
            selected_country
        ),

        selected_continent=(
            selected_continent
        ),

        beer=beer,
        spirit=spirit,
        wine=wine
    )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )