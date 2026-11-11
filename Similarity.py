import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px

MODEL = "similarity_clustering_pipeline_v2"
DATA = "welcome_survey_simple_v2.csv"
NAMES_AND_DESCRIPTIONS = "survey_names_and_descriptions_v2"

@st.cache_data
def get_model():
    return load_model(MODEL)

@st.cache_data
def get_names_and_descriptions():
    with open(NAMES_AND_DESCRIPTIONS, "r", encoding="utf-8") as f:
        return json.loads(f.read())

model = get_model()

@st.cache_data
def get_participants_info():
    all_df = pd.read_csv(DATA, sep=";")
    clustered_all_df = predict_model(model, data=all_df)

    return clustered_all_df

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", "Unknown"])
    edu_level = st.selectbox("Wykształcenie", ["Podstawowe", "Średnie", "Wyższe"])
    fav_animals = st.selectbox("Ulubione zwierzęta", ["Psy", "Koty", "Koty i Psy", "Inne", "Brak ulubionych"])
    fav_place = st.selectbox("Ulubione miejsce", ["W górach", "Nad wodą", "W lesie", "Inne"])
    gender = st.radio("Płeć", ["Mężczyzna", "Kobieta"])

choices_df = pd.DataFrame([
          {
                "age": age,
                "edu_level": edu_level,
                "fav_animals": fav_animals,
                "fav_place": fav_place,
                "gender": gender
          }
])

clustered_all_df = get_participants_info()
names_and_descriptions = get_names_and_descriptions()

cluster_id = predict_model(model, data=choices_df)["Cluster"].values[0]
names_and_descriptions_info = names_and_descriptions[cluster_id]
your_cluster_df = clustered_all_df[clustered_all_df["Cluster"] == cluster_id]
len_cluster = len(your_cluster_df)

st.header(f"Należysz do {len_cluster} osobowej grupy: {names_and_descriptions_info['name']}")
st.markdown(names_and_descriptions_info['description'])

col1, col2 = st.columns(2)

with col1:
    most_common_animal = your_cluster_df["fav_animals"].mode()[0]
    animal_count = (your_cluster_df["fav_animals"] == most_common_animal).sum()
    animal_pct = round(animal_count / len_cluster * 100)
    if most_common_animal == "Brak ulubionych":
            most_common_animal = "Brak"
    st.metric("Najczęstsze zwierzęta:", f"{most_common_animal} ({animal_pct}%)")
    
with col2:
    most_common_edu = your_cluster_df["edu_level"].mode()[0]
    edu_count = (your_cluster_df["edu_level"] == most_common_edu).sum()
    edu_pct = round(edu_count / len_cluster * 100)
    st.metric("Najczęstsze wykształcenie:", f"{most_common_edu} ({edu_pct}%)")

col3, col4 = st.columns(2)

with col3:
    most_common_gender = your_cluster_df["gender"].mode()[0]
    gender_count = (your_cluster_df["gender"] == most_common_gender).sum()
    gender_pct = round(gender_count / len_cluster * 100)
    st.metric("Najczęstsza płeć:", f"{most_common_gender} ({gender_pct}%)")

with col4:
    most_common_age = your_cluster_df["age"].mode()[0]
    age_count = (your_cluster_df["age"] == most_common_age).sum()
    age_pct = round(age_count / len_cluster * 100)
    st.metric("Najczęstszy przedział wieku:", f"{most_common_age} ({age_pct}%)")

plot_df = your_cluster_df.copy()
plot_df["fav_place"] = plot_df["fav_place"].astype(str)
plot_df["fav_place"] = plot_df["fav_place"].replace("nan", "Brak")

tab1, tab2, tab3, tab4 = st.tabs(["Kto tu jest", "Obraz grupy", "W procentach", "Na tle wszystkich"])

with tab1:
    st.header("Oto spis członków Twojej grupy:")
    st.dataframe(your_cluster_df, hide_index=True)

with tab2:
    hist = st.selectbox("Co chcesz zobaczyć?",
                         ["Wiek", "Wykształcenie", "Ulubione zwierzęta", "Ulubione miejsca", "Płeć"])
    if hist == "Wiek":
        fig = px.histogram(your_cluster_df.sort_values("age"), x="age")
        fig.update_layout(
            title="Rozkład wieku w grupie",
            xaxis_title="Wiek",
            yaxis_title="Liczba osób"
        )
        st.plotly_chart(fig, width="stretch")

    elif hist == "Wykształcenie":
        fig = px.histogram(your_cluster_df, x="edu_level")
        fig.update_layout(
            title="Rozkład poziomu wykształcenia w grupie",
            xaxis_title="Poziom wykształcenia",
            yaxis_title="Liczba osób"
        )
        st.plotly_chart(fig, width="stretch")

    elif hist == "Ulubione zwierzęta":
        fig = px.histogram(your_cluster_df, x="fav_animals")
        fig.update_layout(
            title="Rozkład ulubionych zwierząt w grupie",
            xaxis_title="Ulubione zwierzęta",
            yaxis_title="Liczba osób"
        )
        st.plotly_chart(fig, width="stretch")

    elif hist == "Ulubione miejsca":
        fig = px.histogram(plot_df, x="fav_place")
        fig.update_layout(
            title="Rozkład ulubionych miejsc w grupie",
            xaxis_title="Ulubione miejsca",
            yaxis_title="Liczba osób"
        )
        st.plotly_chart(fig, width="stretch")

    elif hist == "Płeć":
        fig = px.histogram(your_cluster_df, x="gender")
        fig.update_layout(
            title="Rozkład płci w grupie",
            xaxis_title="Płeć",
            yaxis_title="Liczba osób"
        )
        st.plotly_chart(fig, width="stretch")

with tab3:
    pie = st.selectbox("Co chcesz zobaczyć?",
                       ["Wiek(%)", "Wykształcenie(%)", "Ulubione zwierzęta(%)", "Ulubione miejsca(%)", "Płeć(%)"])
    if pie == "Wiek(%)":
        fig = px.pie(your_cluster_df, names="age", title="Rozkład procentowy wieku w grupie")
        fig.update_traces(textinfo="percent+label", textposition="outside", textfont_size=16)
        fig.update_layout(title_font_size=26, legend_font_size=16)
        st.plotly_chart(fig, width="stretch")

    elif pie == "Wykształcenie(%)":
        fig = px.pie(your_cluster_df, names="edu_level", title="Rozkład procentowy wykształcenia w grupie")
        fig.update_traces(textinfo="percent+label", textposition="outside", textfont_size=16)
        fig.update_layout(title_font_size=26, legend_font_size=16)
        st.plotly_chart(fig, width="stretch")

    elif pie == "Ulubione zwierzęta(%)":
        fig = px.pie(your_cluster_df, names="fav_animals", title="Rozkład procentowy ulubionych zwierząt w grupie")
        fig.update_traces(textinfo="percent+label", textposition="outside", textfont_size=16)
        fig.update_layout(title_font_size=26, legend_font_size=16)
        st.plotly_chart(fig, width="stretch")

    elif pie == "Ulubione miejsca(%)":
        fig = px.pie(plot_df, names="fav_place", title="Rozkład procentowy ulubionych miejsc w grupie")
        fig.update_traces(textinfo="percent+label", textposition="outside", textfont_size=16)
        fig.update_layout(title_font_size=26, legend_font_size=16)
        st.plotly_chart(fig, width="stretch")

    elif pie == "Płeć(%)":
        gender_df = your_cluster_df.copy()
        gender_df["gender"] = gender_df["gender"].astype(str)
        gender_df["gender"] = gender_df["gender"].replace("nan", "Nie podano")
        fig = px.pie(gender_df, names="gender", title="Rozkład procentowy płci w grupie")
        fig.update_traces(textinfo="percent+label", textposition="outside", textfont_size=16)
        fig.update_layout(title_font_size=26, legend_font_size=16)
        st.plotly_chart(fig, width="stretch")

with tab4:
    len_all = len(clustered_all_df)
    cluster_pct = len_cluster / len_all * 100
    pozostali = len_all - len_cluster

    st.markdown("### Twoja grupa")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Osób w grupie:", len_cluster)
    with col2:
        st.metric("Wszystkich:", len_all)
    with col3:
        st.metric("Udział:", f"{cluster_pct:.2f}%")

    fig = px.pie(
        names=["Twoja grupa", "Pozostali"],
        values=[len_cluster, pozostali],
        title="Udział Twojej grupy",
        hole=0.5,
        color_discrete_sequence=["#0C3F21", "#EF553B"]
    )
    fig.update_traces(textinfo="percent+label", textfont_size=20, textposition="outside", texttemplate="%{label}<br>%{percent:.2%}")
    fig.update_layout(showlegend=False, height=500, title_font_size=26)
    st.plotly_chart(fig, width="stretch")