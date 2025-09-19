import os
import json
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
print("Loaded SERPAPI_KEY:", SERPAPI_KEY)
load_dotenv()
CURRENCY_KEY = os.getenv("CURRENCY_KEY")
print("Loaded CURRENCY_KEY:", CURRENCY_KEY)



st.set_page_config(page_title="Travel Agent QA", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_hotel_data(csv_path="hotel_data.csv"):
    try:
        df = pd.read_csv(csv_path)
      
        required = {"hotel_name","city","price_per_night","rating"}
        if not required.issubset(set(df.columns)):
            st.warning(f"hotel_data.csv must contain columns: {required}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load hotel data: {e}")
        return pd.DataFrame()
    

ALLOWED_COUNTRIES = ["France", "UK", "Japan", "China", "UAE", "Pakistan"]


@st.cache_data
def load_flights_data(csv_path="flights_data.csv"):
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def book_flight(df, flight_number, passengers, save_path="flights_data.csv"):
    row_index = df[df["flight_number"] == flight_number].index
    if not row_index.empty:
        idx = row_index[0]
        if df.at[idx, "seats_available"] >= passengers:
            df.at[idx, "seats_available"] -= passengers
            df.to_csv(save_path, index=False)   
            return True, df
        else:
            return False, df
    return False, df



def recommend_hotels(df, city, budget_per_night, rating, location, top_k=5):
    """
    Simple content-based recommender:
    - filter by city
    - optional budget filter (price_per_night <= budget_per_night)
    - rank by combined score: rating normalized - price penalty
    """
    if df.empty:
        return []

   
    df_city = df[df["city"].str.lower() == city.lower()].copy()
    if df_city.empty:
        
        df_city = df[df["city"].str.lower().str.contains(city.lower())].copy()

    if df_city.empty:
        return []

    
    
    df_city["rating_norm"] = df_city["rating"] / 5.0
    
    max_price = df_city["price_per_night"].max() if df_city["price_per_night"].max() > 0 else 1.0
    df_city["price_norm"] = df_city["price_per_night"] / max_price

    
    price_penalty = 0.6

    
    df_city["score"] = df_city["rating_norm"] - price_penalty * df_city["price_norm"]

    
    if budget_per_night is not None:
        df_city["within_budget"] = df_city["price_per_night"] <= budget_per_night
        
        df_city.loc[df_city["within_budget"], "score"] += 0.15

    
    df_city = df_city.sort_values(by=["score","rating"], ascending=[False, False])
    
    recs = df_city.head(top_k).to_dict(orient="records")
    return recs

@st.cache_resource(show_spinner=False)
def load_json_dataset(json_path="travel_agent_squad.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    titles = []
    for item in raw.get("data", []):
        title = item.get("title", "")
        titles.append(title)
        for para in item.get("paragraphs", []):
            context = para.get("context", "")
            rows.append({"title": title, "context": context})

    dedup = {}
    for r in rows:
        if r["title"] not in dedup:
            dedup[r["title"]] = r["context"]
    contexts = [{"title": t, "context": dedup[t]} for t in dedup]
    return raw, contexts


def smart_select_context(question, contexts):
    q_lower = question.lower()
    for c in contexts:
        if c["title"].lower() in q_lower:
            return c["context"], c["title"]

    q_words = set([w.strip(".,?!") for w in q_lower.split() if len(w) > 2])
    best = None
    best_score = 0
    for c in contexts:
        c_words = set([w.strip(".,?!") for w in c["context"].lower().split() if len(w) > 2])
        score = len(q_words & c_words)
        if score > best_score:
            best_score = score
            best = c
    if best:
        return best["context"], best["title"]

    return " ".join([c["context"] for c in contexts]), "All"


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path_or_name):
    device = 0 if torch.cuda.is_available() else -1
    qa = pipeline("question-answering", model=model_path_or_name, tokenizer=model_path_or_name, device=device)
    return qa


def google_search(query, num_results=3):
    if not SERPAPI_KEY:
        return [{"title": "⚠️ SERPAPI_KEY missing", "snippet": "Add your API key in .env", "link": "https://serpapi.com/"}]

    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        results = []
        for r in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", "")
            })
        return results
    except Exception as e:
        return [{"title": "Error", "snippet": str(e), "link": ""}]


st.markdown("""
<style>
.reportview-container .main .block-container{padding-top:1rem;}
.chat-box {border-radius:12px;padding:12px;margin-bottom:8px;}
.user {background-color:#e8f0ff;padding:10px;border-radius:10px;}
.bot {background-color:#f1f7ed;padding:10px;border-radius:10px;}
.small-muted {color:#6c757d;font-size:12px;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Data & Model")
    json_path = st.text_input("JSON file path", "travel_agent_squad.json")
    raw_data, contexts = load_json_dataset(json_path)
    st.write(f"Loaded {len(contexts)} country contexts.")
    st.markdown("**Available countries:**")
    st.write(", ".join([c["title"] for c in contexts]))

    st.markdown("---")
    chosen_model = st.selectbox("Choose model", options=["distilbert-base-uncased-distilled-squad", "distilbert-base-uncased"], index=0)
    show_context = st.checkbox("Show selected context", value=True)

    st.markdown("---")
    st.header("Hotel Price Prediction (ML)")
    st.info("Upload a CSV with columns: guests, nights, price")

    uploaded_csv = st.file_uploader("Upload hotel_booking.csv", type="csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("CSV Preview:", df.head())

        if {"guests", "nights", "price"}.issubset(df.columns):
            X = df[["guests", "nights"]]
            y = df["price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.success(f"Model Trained  (MSE: {mse:.2f})")

            st.subheader(" Make a Prediction")
            g = st.number_input("Number of guests", min_value=1, max_value=10, value=2)
            n = st.number_input("Number of nights", min_value=1, max_value=30, value=3)
            if st.button("Predict Hotel Price"):
                price_pred = model.predict(np.array([[g, n]]))[0]
                st.success(f"Predicted Price: ${price_pred:.2f}")
        else:
            st.warning("CSV must have columns: guests, nights, price")
    else:
        st.warning("Please upload hotel_booking.csv to enable prediction.")

        st.markdown("---")



st.set_page_config(page_title=" Flight Booking", layout="wide")

st.title("🛫 Flight Booking System")


uploaded_csv = st.file_uploader("Upload your flights_data.csv", type=["csv"])

if uploaded_csv:
    flights_df = pd.read_csv(uploaded_csv)
    st.success(" Using uploaded flights data")
else:
    flights_df = load_flights_data("flights_data.csv")
    st.info(" Using default flights_data.csv")

if not flights_df.empty:
    st.sidebar.header(" Search Flights")

    dep = st.sidebar.selectbox("Departure", ALLOWED_COUNTRIES)
    dest = st.sidebar.selectbox("Destination", ALLOWED_COUNTRIES)
    travel_date = st.sidebar.date_input("Travel Date", datetime.date.today())
    passengers = st.sidebar.number_input("Passengers", min_value=1, max_value=10, value=1)

    if dep == dest:
        st.sidebar.error(" Departure and Destination cannot be the same.")
    else:
        search = st.sidebar.button("Search Flights")

        if search:
            results = flights_df[
                (flights_df["departure"].str.lower() == dep.lower()) &
                (flights_df["destination"].str.lower() == dest.lower()) &
                (flights_df["date"] == str(travel_date))
            ]

            if results.empty:
                st.warning(" No flights found for this route and date.")
            else:
                st.subheader(f"Available Flights: {dep} → {dest} on {travel_date}")
                for _, row in results.iterrows():
                    st.markdown(f"""
                    **{row['airline']} {row['flight_number']}**
                    - Price: ${row['price']}
                    - Seats Available: {row['seats_available']}
                    """)

                    if st.button(f"Book {row['flight_number']}", key=row["flight_number"]):
                        save_path = uploaded_csv if uploaded_csv else "flights_data.csv"
                        success, flights_df = book_flight(flights_df, row["flight_number"], passengers, save_path)
                        if success:
                            st.success(f"🎉 Booking confirmed for {passengers} passenger(s) on {row['airline']} {row['flight_number']}")
                        else:
                            st.error(" Not enough seats available.")
else:
    st.error(" No flight data available. Please upload flights_data.csv.")




qa_pipeline = load_pipeline(chosen_model)


st.title("✈️ Travel Agent QA — Hybrid (Dataset + Google)")
st.write("Ask travel-related questions (France, UK, Japan, China, UAE...). If dataset has no good answer, fallback to Google Search.")

st.subheader("Chat")
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([4,1])
with col1:
    user_question = st.text_input("Ask a question...", key="q_in", placeholder="e.g. What is the capital of France?")
with col2:
    if st.button("Send"):
        if user_question.strip():
            st.session_state.history.append({"role": "user", "text": user_question.strip()})
        else:
            st.warning("Please type a question first.")


if st.session_state.history and st.session_state.history[-1]["role"] == "user":
    q = st.session_state.history[-1]["text"]
    context, chosen_title = smart_select_context(q, contexts)

    with st.spinner(f"Finding answer from '{chosen_title}'..."):
        try:
            res = qa_pipeline(question=q, context=context)
            answer_text = res.get("answer", "")
            score = res.get("score", 0.0)
        except Exception as e:
            answer_text = f"Error: {e}"
            score = 0.0

    if not answer_text or score < 0.3:
        google_results = google_search(q)
        st.session_state.history.append({"role": "bot", "text": "Google Search Results:", "title": "Google", "score": 1.0, "google": google_results})
    else:
        st.session_state.history.append({"role": "bot", "text": answer_text, "title": chosen_title, "score": score, "context": context})


for msg in st.session_state.history[-20:]:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-box user'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        if msg.get("title") == "Google":
            st.markdown(f"<div class='chat-box bot'><b>Bot (Google):</b> {msg['text']}</div>", unsafe_allow_html=True)
            for r in msg.get("google", []):
                st.markdown(f"- [{r['title']}]({r['link']}) — {r['snippet']}")
        else:
            st.markdown(f"<div class='chat-box bot'><b>Bot ({msg.get('title','')}):</b> {msg['text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'>Confidence: {msg.get('score',0.0):.2f}</div>", unsafe_allow_html=True)
            if show_context:
                ans = msg["text"]
                ctx = msg.get("context", "")
                if ans and ans in ctx:
                    highlighted = ctx.replace(ans, f"**{ans}**")
                else:
                    highlighted = ctx[:500] + ("..." if len(ctx) > 500 else "")
                st.markdown(f"<div class='small-muted'>Context ({msg.get('title','')}): {highlighted}</div>", unsafe_allow_html=True)



st.header(" Hotel Recommendation")


hotel_df = load_hotel_data("hotel_data.csv") 

if hotel_df.empty:
    st.info("No hotel_data.csv found in project folder. Upload / save sample file or change path.")
    
    uploaded_hotels = st.file_uploader("Or upload hotel_data.csv", type=["csv"], key="hotel_rec_upload")
    if uploaded_hotels:
        hotel_df = pd.read_csv(uploaded_hotels)
        st.success("Hotel data uploaded.")
else:
    st.write(f"Loaded {len(hotel_df)} hotels.")


rec_city = st.text_input("City for recommendation", value="Paris")
rec_budget_total = st.number_input("Budget (total) — optional (USD)", min_value=0, value=300)
rec_budget_per_night = None
rec_guests = st.number_input("Guests", min_value=1, max_value=10, value=2)
rec_nights = st.number_input("Nights", min_value=1, max_value=7, value=3)
if rec_budget_total and rec_nights > 0:
    rec_budget_per_night = rec_budget_total / rec_nights

top_k = st.slider("How many hotels to show", min_value=1, max_value=10, value=3)

if st.button("Recommend Hotels"):
    recs = recommend_hotels(hotel_df, rec_city, rec_budget_per_night, rec_guests, rec_nights, top_k=top_k)
    if not recs:
        st.warning("No recommendations found — try different city or upload hotel_data.csv.")
    else:
        st.markdown(f"### Top {len(recs)} hotels in {rec_city}")
        for r in recs:
            st.markdown(f"**{r['hotel_name']}** — {r['location'] if 'location' in r else ''}")
            st.markdown(f"- Price / night: ${r['price_per_night']:.2f} — Rating: {r['rating']}/5")
            if 'link' in r and pd.notna(r['link']):
                st.markdown(f"- [Book / Details]({r['link']})")
            st.markdown(f"- Score: {r.get('score',0):.3f}")
            st.markdown("---")

            # ---------------------- Currency Converter Section ----------------------
st.markdown("---")
st.header(" Currency Converter")

API_KEY = os.getenv("CURRENCY_KEY")  

if not API_KEY:
    st.error("API key not found. Create a .env file with: CURRENCY_KEY=your_api_key")
else:
    amount = st.number_input("Enter amount:", min_value=0.01, value=100.0, step=1.0, format="%.2f", key="curr_amount")
    from_currency = st.text_input("From Currency (e.g., USD):", "USD", key="curr_from").strip().upper()
    to_currency = st.text_input("To Currency (e.g., PKR):", "PKR", key="curr_to").strip().upper()

    def convert_currency(amount, from_currency, to_currency, api_key, debug=False):
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{from_currency}"
        try:
            r = requests.get(url, timeout=10)
        except Exception as e:
            return None, None, f"Request failed: {e}"

        if r.status_code != 200:
            return None, None, f"API returned status {r.status_code}: {r.text[:300]}"

        data = r.json()
        rate = None
        if isinstance(data, dict):
            if "conversion_rates" in data and isinstance(data["conversion_rates"], dict):
                rate = data["conversion_rates"].get(to_currency)
            elif "rates" in data and isinstance(data["rates"], dict):
                rate = data["rates"].get(to_currency)

        if rate is None:
            if debug:
                st.write("Debug - API JSON keys:", list(data.keys()))
            return None, None, f"Rate for {to_currency} not found in API response."

        converted = round(amount * rate, 2)
        return converted, rate, None

    if st.button("Convert", key="curr_btn"):
        converted, rate, err = convert_currency(amount, from_currency, to_currency, API_KEY, debug=True)
        if err:
            st.error("Conversion failed: " + err)
        else:
            st.success(f"{amount} {from_currency} = {converted} {to_currency}  (1 {from_currency} = {rate})")



st.markdown("---")
st.write("Tips:")
st.markdown("- Mention the country name for more accurate results.")
st.markdown("- If dataset confidence is low, bot auto-switches to **Google Search**.")
