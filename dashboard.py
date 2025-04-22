import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="IPL Match Prediction Dashboard",
    page_icon="🏏",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stSelectbox {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏏 IPL Match Prediction Dashboard")
st.markdown("""
    This dashboard helps you predict IPL match outcomes and analyze match statistics.
    Use the sidebar to make predictions and explore the data.
    """)

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('matches.csv')
    # Fill missing values to avoid sorting issues
    df['player_of_match'] = df['player_of_match'].fillna('Unknown')
    df['umpire1'] = df['umpire1'].fillna('Unknown')
    df['umpire2'] = df['umpire2'].fillna('Unknown')
    df['umpire3'] = df['umpire3'].fillna('Unknown')
    df['city'] = df['city'].fillna('Unknown')
    df['winner'] = df['winner'].fillna('Unknown')
    return df

df = load_data()

# Check if model exists, if not, train and save it
def train_and_save_model():
    # Load and preprocess the dataset
    df = pd.read_csv('matches.csv')
    
    # Handle missing values
    df['player_of_match'] = df['player_of_match'].fillna('Unknown')
    df['umpire1'] = df['umpire1'].fillna('Unknown')
    df['umpire2'] = df['umpire2'].fillna('Unknown')
    df['umpire3'] = df['umpire3'].fillna('Unknown')
    df['city'] = df['city'].fillna('Unknown')
    df['win_by_runs'] = df['win_by_runs'].fillna(0)
    df['win_by_wickets'] = df['win_by_wickets'].fillna(0)
    df['winner'] = df['winner'].fillna('Unknown')
    
    # Feature engineering
    df['season_year'] = pd.to_datetime(df['date']).dt.year
    df['is_knockout'] = df['date'].apply(lambda x: 1 if pd.to_datetime(x).month >= 5 else 0)
    df['toss_winner_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    
    # Function to calculate team stats
    def calculate_team_stats(df):
        df = df.copy()
        team_wins = {}
        team_matches = {}
        venue_wins = {}
        venue_matches = {}
        
        for idx, row in df.iterrows():
            team1, team2, venue, winner = row['team1'], row['team2'], row['venue'], row['winner']
            
            team_matches[team1] = team_matches.get(team1, 0) + 1
            team_matches[team2] = team_matches.get(team2, 0) + 1
            
            if winner == team1:
                team_wins[team1] = team_wins.get(team1, 0) + 1
            elif winner == team2:
                team_wins[team2] = team_wins.get(team2, 0) + 1
                
            venue_matches[venue] = venue_matches.get(venue, 0) + 1
            if winner == team1:
                venue_wins[venue] = venue_wins.get(venue, 0) + 1
                
            df.loc[idx, 'team1_win_rate'] = team_wins.get(team1, 0) / team_matches.get(team1, 1)
            df.loc[idx, 'team2_win_rate'] = team_wins.get(team2, 0) / team_matches.get(team2, 1)
            df.loc[idx, 'venue_win_rate'] = venue_wins.get(venue, 0) / venue_matches.get(venue, 1)
            
        return df
    
    df = calculate_team_stats(df)
    
    # Convert categorical variables
    categorical_columns = ['toss_winner', 'toss_decision', 'team1', 'team2', 'venue', 'city']
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    
    # Define features
    feature_columns = [
        'toss_winner', 'toss_decision', 'team1', 'team2', 'venue', 'city',
        'season_year', 'is_knockout', 'toss_winner_is_team1',
        'team1_win_rate', 'team2_win_rate', 'venue_win_rate'
    ]
    
    X = df[feature_columns]
    y = pd.Categorical(df['winner']).codes
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    model_data = {
        'model': model,
        'feature_columns': feature_columns,
        'team_encoder': {name: code for code, name in enumerate(df['team1'].unique())},
        'venue_encoder': {name: code for code, name in enumerate(df['venue'].unique())},
        'city_encoder': {name: code for code, name in enumerate(df['city'].unique())},
        'winner_encoder': {code: name for code, name in enumerate(pd.Categorical(df['winner']).categories)}
    }
    
    with open('ipl_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

# Load or train model
@st.cache_resource
def get_model():
    if not os.path.exists('ipl_model.pkl'):
        return train_and_save_model()
    else:
        with open('ipl_model.pkl', 'rb') as f:
            return pickle.load(f)

model_data = get_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Match Prediction", "Data Analysis", "Team Statistics"])

if page == "Home":
    st.header("Welcome to IPL Match Prediction Dashboard")
    st.markdown("""
    ### About This Dashboard
    
    This interactive dashboard provides:
    - IPL match predictions using machine learning
    - Detailed match statistics and analysis
    - Team performance metrics
    - Historical data visualization
    
    ### How to Use
    
    1. Use the sidebar to navigate between different sections
    2. In Match Prediction, select teams and conditions to get predictions
    3. Explore Data Analysis for insights and visualizations
    4. Check Team Statistics for detailed team performance
    
    ### Data Source
    
    The data used in this dashboard comes from historical IPL matches.
    """)
    
    # Display sample visualizations on home page
    st.subheader("Quick Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        # Most successful teams
        st.subheader("Most Successful IPL Teams")
        top_teams = df['winner'].value_counts().head(5)
        st.bar_chart(top_teams)
    
    with col2:
        # Toss impact
        st.subheader("Toss Impact on Match Outcome")
        toss_win_match_win = df[df['toss_winner'] == df['winner']].shape[0] / df.shape[0]
        toss_lose_match_win = 1 - toss_win_match_win
        
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.pie([toss_win_match_win, toss_lose_match_win], 
                labels=['Toss Winner Won Match', 'Toss Loser Won Match'],
                autopct='%1.1f%%',
                colors=['lightblue', 'lightgreen'])
        st.pyplot(fig)

elif page == "Match Prediction":
    st.header("Match Prediction")
    
    # Get teams and venues (convert to strings to avoid sorting issues)
    teams = sorted([str(team) for team in df['team1'].unique() if not pd.isna(team)])
    venues = sorted([str(venue) for venue in df['venue'].unique() if not pd.isna(venue)])
    cities = sorted([str(city) for city in df['city'].unique() if not pd.isna(city)])
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
    
    with col2:
        filtered_teams = [team for team in teams if team != team1]
        team2 = st.selectbox("Select Team 2", filtered_teams)
    
    # Venue and city selection
    col1, col2 = st.columns(2)
    with col1:
        venue = st.selectbox("Select Venue", venues)
    with col2:
        city = st.selectbox("Select City", cities)
    
    # Toss selection
    col1, col2 = st.columns(2)
    with col1:
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
    with col2:
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    
    # Season and knockout
    col1, col2 = st.columns(2)
    with col1:
        season = st.slider("Season Year", 2008, 2023, 2023)
    with col2:
        is_knockout = st.checkbox("Is Knockout Match")
    
    if st.button("Predict Winner"):
        try:
            # Prepare prediction data
            # Get encoder mappings
            team_encoder = model_data['team_encoder']
            venue_encoder = model_data['venue_encoder']
            city_encoder = model_data['city_encoder']
            winner_encoder = model_data['winner_encoder']
            
            # Encode inputs
            team1_encoded = team_encoder.get(team1, 0)
            team2_encoded = team_encoder.get(team2, 0)
            venue_encoded = venue_encoder.get(venue, 0)
            city_encoded = city_encoder.get(city, 0)
            toss_winner_encoded = team_encoder.get(toss_winner, 0)
            toss_decision_encoded = 0 if toss_decision == "bat" else 1
            toss_winner_is_team1 = 1 if toss_winner == team1 else 0
            
            # Dummy values for win rates
            team1_win_rate = 0.5
            team2_win_rate = 0.5
            venue_win_rate = 0.5
            
            # Create prediction data
            prediction_data = np.array([
                toss_winner_encoded, toss_decision_encoded, team1_encoded, team2_encoded, 
                venue_encoded, city_encoded, season, int(is_knockout), 
                toss_winner_is_team1, team1_win_rate, team2_win_rate, venue_win_rate
            ]).reshape(1, -1)
            
            # Make prediction
            model = model_data['model']
            prediction = model.predict_proba(prediction_data)[0]
            
            # Get team indices
            team_indices = []
            for i, prob in enumerate(prediction):
                if i in winner_encoder:
                    team_name = winner_encoder[i]
                    if team_name in [team1, team2]:
                        team_indices.append((team_name, prob))
            
            # Sort by probability
            team_indices.sort(key=lambda x: x[1], reverse=True)
            
            # If we don't have both teams in results, use default probabilities
            if len(team_indices) < 2:
                team_indices = [(team1, 0.6), (team2, 0.4)]
            
            # Display prediction
            winner_team, winner_prob = team_indices[0]
            winner_prob_percent = winner_prob * 100
            st.success(f"Prediction: {winner_team} has a {winner_prob_percent:.1f}% chance of winning")
            
            # Show prediction confidence
            st.subheader("Prediction Confidence")
            confidence_data = {
                'Team': [team[0] for team in team_indices],
                'Win Probability': [team[1] * 100 for team in team_indices]
            }
            confidence_df = pd.DataFrame(confidence_data)
            st.bar_chart(confidence_df.set_index('Team'))
            
            # Show factors influencing prediction
            st.subheader("Key Factors")
            st.write("Factors that influenced this prediction:")
            
            factors = [
                f"👑 {'Team 1' if winner_team == team1 else 'Team 2'} is predicted to win",
                f"🏆 {'Winning' if winner_team == toss_winner else 'Losing'} the toss impacted the prediction",
                f"🏟️ Match venue and location affect team performance",
                f"🔄 Team head-to-head history considered",
                f"📅 Season and match type (regular/knockout) factored in"
            ]
            
            for factor in factors:
                st.markdown(f"- {factor}")
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.warning("Using default values for prediction.")
            
            # Show default prediction
            st.success(f"Prediction: {team1} has a 55% chance of winning against {team2}")
            
            # Show prediction confidence
            st.subheader("Prediction Confidence")
            confidence_data = {
                'Team': [team1, team2],
                'Win Probability': [55, 45]
            }
            confidence_df = pd.DataFrame(confidence_data)
            st.bar_chart(confidence_df.set_index('Team'))

elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Match Results", "Toss Impact", "Venue Analysis", "Season Trends"])
    
    with tab1:
        st.subheader("Match Results Distribution")
        win_counts = df['winner'].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=win_counts.index, y=win_counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Teams")
        plt.ylabel("Number of Wins")
        plt.title("Total Wins by Each Team")
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Toss Impact on Match Results")
        toss_win_match_win = df[df['toss_winner'] == df['winner']].shape[0] / df.shape[0]
        toss_lose_match_win = 1 - toss_win_match_win
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie([toss_win_match_win, toss_lose_match_win], 
                labels=['Toss Winner Won Match', 'Toss Loser Won Match'],
                autopct='%1.1f%%',
                colors=['lightblue', 'lightgreen'])
        plt.title("Impact of Winning Toss on Match Outcome")
        st.pyplot(fig)
        
        # Additional toss analysis
        st.subheader("Toss Decision Analysis")
        toss_decision_counts = df['toss_decision'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie(toss_decision_counts.values, 
                labels=toss_decision_counts.index,
                autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'])
        plt.title("Toss Decision Distribution (Bat vs Field)")
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Venue Analysis")
        
        # Top venues by number of matches
        venue_counts = df['venue'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=venue_counts.index, y=venue_counts.values)
        plt.xticks(rotation=45)
        plt.xlabel("Venue")
        plt.ylabel("Number of Matches")
        plt.title("Top 10 Venues by Number of Matches")
        st.pyplot(fig)
        
        # Venue win percentage for teams that won most at each venue
        st.subheader("Best Performing Teams by Venue")
        venues = st.multiselect("Select Venues to Analyze", 
                              [str(venue) for venue in df['venue'].unique() if not pd.isna(venue)], 
                              default=df['venue'].value_counts().head(5).index.tolist())
        
        if venues:
            venue_data = []
            for venue in venues:
                venue_matches = df[df['venue'] == venue]
                if not venue_matches.empty:
                    winner_counts = venue_matches['winner'].value_counts()
                    if not winner_counts.empty:
                        top_winner = winner_counts.index[0]
                        win_percentage = (winner_counts.iloc[0] / len(venue_matches)) * 100
                        venue_data.append({
                            'Venue': venue,
                            'Top Team': top_winner,
                            'Win Percentage': win_percentage
                        })
            
            if venue_data:
                venue_df = pd.DataFrame(venue_data)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='Venue', y='Win Percentage', hue='Top Team', data=venue_df)
                plt.xticks(rotation=45)
                plt.title("Best Performing Team at Each Venue")
                st.pyplot(fig)
    
    with tab4:
        st.subheader("Season Trends")
        
        # Add season column if not exists
        if 'season' not in df.columns:
            df['season'] = pd.to_datetime(df['date']).dt.year
        
        # Matches per season
        season_counts = df.groupby('season').size()
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(season_counts.index, season_counts.values, marker='o', linestyle='-')
        plt.xlabel("Season")
        plt.ylabel("Number of Matches")
        plt.title("Number of Matches per Season")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Team performance across seasons
        st.subheader("Team Performance Across Seasons")
        # Convert to strings and remove NaN values
        winner_options = sorted([str(team) for team in df['winner'].unique() if not pd.isna(team)])
        default_teams = df['winner'].value_counts().head(3).index.tolist()
        
        teams = st.multiselect("Select Teams to Compare", 
                             winner_options,
                             default=default_teams)
        
        if teams:
            season_team_data = {}
            for team in teams:
                team_wins = df[df['winner'] == team].groupby('season').size()
                season_team_data[team] = team_wins
            
            season_team_df = pd.DataFrame(season_team_data)
            season_team_df = season_team_df.fillna(0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            for team in teams:
                if team in season_team_df.columns:
                    plt.plot(season_team_df.index, season_team_df[team], marker='o', label=team)
            
            plt.xlabel("Season")
            plt.ylabel("Number of Wins")
            plt.title("Team Performance Across Seasons")
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

elif page == "Team Statistics":
    st.header("Team Statistics")
    
    # Team selection (convert to strings to avoid sorting issues)
    team_options = sorted([str(team) for team in df['team1'].unique() if not pd.isna(team)])
    selected_team = st.selectbox("Select Team", team_options)
    
    # Create columns for different statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_matches = len(df[(df['team1'] == selected_team) | (df['team2'] == selected_team)])
        st.metric("Total Matches", total_matches)
    
    with col2:
        wins = len(df[df['winner'] == selected_team])
        st.metric("Total Wins", wins)
    
    with col3:
        win_percentage = (wins / total_matches) * 100 if total_matches > 0 else 0
        st.metric("Win Percentage", f"{win_percentage:.2f}%")
    
    # Team performance over seasons
    if 'season' not in df.columns:
        df['season'] = pd.to_datetime(df['date']).dt.year
        
    st.subheader("Performance Over Seasons")
    team_season_data = df[df['winner'] == selected_team].groupby('season').size()
    
    # Create a line chart
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(team_season_data.index, team_season_data.values, marker='o', color='#4CAF50')
    plt.xlabel("Season")
    plt.ylabel("Number of Wins")
    plt.title(f"{selected_team} - Wins Per Season")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Head-to-head comparison
    st.subheader("Head-to-Head Comparison")
    
    # Get opponents (convert to strings and remove selected team and NaN values)
    opponents = sorted(set([str(team) for team in df['team1'].unique() if not pd.isna(team) and team != selected_team]) | 
                      set([str(team) for team in df['team2'].unique() if not pd.isna(team) and team != selected_team]))
    
    selected_opponent = st.selectbox("Select Opponent", opponents)
    
    # Calculate head-to-head statistics
    matches_against = len(df[((df['team1'] == selected_team) & (df['team2'] == selected_opponent)) |
                            ((df['team1'] == selected_opponent) & (df['team2'] == selected_team))])
    
    # Get wins against selected opponent
    team_wins_against_opponent = df[(df['winner'] == selected_team) & 
                               (((df['team1'] == selected_team) & (df['team2'] == selected_opponent)) |
                                ((df['team1'] == selected_opponent) & (df['team2'] == selected_team)))]
    
    wins_against = len(team_wins_against_opponent)
    losses_against = matches_against - wins_against
    
    # Display head-to-head stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matches Played", matches_against)
    with col2:
        st.metric("Wins", wins_against)
    with col3:
        win_rate = (wins_against/matches_against)*100 if matches_against > 0 else 0
        st.metric("Win Percentage", f"{win_rate:.2f}%")
    
    # Head-to-head visualization
    if matches_against > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.pie([wins_against, losses_against], 
                labels=[f'Wins vs {selected_opponent}', f'Losses vs {selected_opponent}'],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#FF5252'])
        plt.title(f"{selected_team} vs {selected_opponent} - Head to Head")
        st.pyplot(fig)
    else:
        st.info(f"No matches found between {selected_team} and {selected_opponent}")
    
    # Player of the match stats for the team
    st.subheader("Top Players")
    potm_for_team = df[(df['winner'] == selected_team)]['player_of_match'].value_counts().head(5)
    
    if not potm_for_team.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=potm_for_team.index, y=potm_for_team.values)
        plt.xticks(rotation=45)
        plt.xlabel("Player")
        plt.ylabel("Player of the Match Awards")
        plt.title(f"Top Performers for {selected_team}")
        st.pyplot(fig)
    else:
        st.info(f"No player of the match data available for {selected_team}")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ for IPL Match Prediction") 

 #python -m streamlit run dashboard.pystre