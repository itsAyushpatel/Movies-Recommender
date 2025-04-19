import os
import json
import nltk
import numpy as np
import requests
import time
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
DATA_FILE = "movie_data.json"
API_KEY_FILE = "tmdb_api_key.txt"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TARGET_MOVIE_COUNT = 4719  # Updated to 8000 total

# Target distribution
HOLLYWOOD_COUNT = 2500
BOLLYWOOD_COUNT = 1000
SOUTH_INDIAN_COUNT = 500 
WEB_SERIES_COUNT = 500  # Both Hollywood and Bollywood web series

class MovieRecommender:
    def __init__(self):
        self.movies = []
        self.tfidf_matrix = None
        self.vectorizer = None
        self.api_key = self._load_api_key()
        self.unique_movie_ids = set()  # To track unique movies
        
        # Load existing data or fetch new data
        if os.path.exists(DATA_FILE):
            self._load_data()
            # If loaded data is less than target, fetch more
            if len(self.movies) < TARGET_MOVIE_COUNT:
                print(f"Only {len(self.movies)} movies in dataset, fetching more to reach {TARGET_MOVIE_COUNT}...")
                self._fetch_additional_data()
        else:
            self._fetch_and_process_data()
        
        self._prepare_tfidf()
    
    def _load_api_key(self):
        """Load TMDB API key from file"""
        try:
            with open(API_KEY_FILE, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: {API_KEY_FILE} not found. Please create this file with your TMDB API key.")
            return "YOUR_API_KEY_HERE"  # Placeholder for testing
    
    def _load_data(self):
        """Load movie data from JSON file"""
        print("Loading existing movie data...")
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            self.movies = json.load(f)
        # Populate the unique IDs set
        self.unique_movie_ids = set(movie['id'] for movie in self.movies)
        print(f"Loaded {len(self.movies)} movies")
    
    def _fetch_and_process_data(self):
        """Fetch a large dataset of movies from TMDB API using multiple methods"""
        print(f"Fetching {TARGET_MOVIE_COUNT} movies from TMDB API...")
        self.movies = []
        self.unique_movie_ids = set()
        
        # Get list of all available genres
        genres = self._get_genres()
        
        # Track progress
        start_time = time.time()
        
        # Track movie counts by category
        hollywood_count = 0
        bollywood_count = 0
        south_indian_count = 0
        web_series_count = 0
        
        # 1. Fetch Hollywood/International movies first (popular movies)
        print(f"Phase 1: Fetching Hollywood/International movies (target: {HOLLYWOOD_COUNT})...")
        
        # 1.1 Fetch popular movies
        self._fetch_from_endpoint("movie/popular", pages=20)
        hollywood_count = len(self.unique_movie_ids)
        self._save_progress("Popular Movies")
        
        # 1.2 Fetch top-rated movies
        self._fetch_from_endpoint("movie/top_rated", pages=20)
        hollywood_count = len(self.unique_movie_ids)
        self._save_progress("Top Rated Movies")
        
        # 1.3 Fetch movies by year (recent years first)
        current_year = datetime.now().year
        for year in range(current_year, current_year - 30, -1):  # 30 years back
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
                
            # Make sure we're specifically getting movies from this year
            self._fetch_by_year(year, max_pages=3, strict_year=True)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Year {year}")
        
        # 1.4 Fetch movies by genre
        for genre in genres:
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
            self._fetch_by_genre(genre['id'], genre['name'], max_pages=5)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Genre {genre['name']}")
        
        # 1.5 Fetch by language (focus on English)
        if hollywood_count < HOLLYWOOD_COUNT:
            self._fetch_by_language("en", max_pages=20)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress("English language movies")
        
        # 1.6 Fetch by top studios
        top_studios = [
            420,   # Marvel Studios
            2,     # Disney
            33,    # Universal Pictures
            4,     # Paramount
            174,   # Warner Bros. Pictures
            7505,  # Sony Pictures
            25,    # 20th Century Fox
            4171,  # Pixar
            41,    # Dreamworks
        ]
        
        for studio_id in top_studios:
            if hollywood_count >= HOLLYWOOD_COUNT:
                break
            self._fetch_by_company(studio_id, max_pages=5)
            hollywood_count = len(self.unique_movie_ids)
            self._save_progress(f"Studio {studio_id}")
        
        # 2. Fetch Bollywood movies (Hindi cinema)
        print(f"Phase 2: Fetching Bollywood movies (target: {BOLLYWOOD_COUNT})...")
        bollywood_start_count = len(self.unique_movie_ids)
        bollywood_fetched = 0
        
        # 2.1 Using discover endpoint with Hindi language parameter
        page = 1
        while bollywood_fetched < BOLLYWOOD_COUNT:
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language=hi&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                        
                    before_count = len(self.unique_movie_ids)
                    self._process_movie_results(results, is_bollywood=True)
                    after_count = len(self.unique_movie_ids)
                    bollywood_fetched += (after_count - before_count)
                        
                    self._save_progress(f"Bollywood movies (page {page})")
                    page += 1
                    time.sleep(0.5)  # Prevent rate limiting
                        
                    # Break if we've reached end of results
                    if page > data.get('total_pages', 1) or page > 100:  # Allow up to 100 pages
                        break
                else:
                    print(f"Error {response.status_code} when fetching Bollywood movies")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching Bollywood movies: {e}")
                time.sleep(2)
                
        print(f"Fetched {bollywood_fetched} Bollywood movies")
        
        # 2.2 Fetch by popular Bollywood studios/production companies
        if bollywood_fetched < BOLLYWOOD_COUNT:
            bollywood_studios = [
                1569,   # Yash Raj Films
                2515,   # Dharma Productions
                1913,   # Excel Entertainment
                5626,   # Red Chillies Entertainment
                1884,   # UTV Motion Pictures
                3538,   # T-Series
                7294,   # Viacom18 Studios
                128250, # Aamir Khan Productions
                156782  # Sanjay Leela Bhansali Productions
            ]
            
            for studio_id in bollywood_studios:
                if bollywood_fetched >= BOLLYWOOD_COUNT:
                    break
                    
                # Use a more targeted approach that combines company and language
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={studio_id}&with_original_language=hi&page=1&sort_by=popularity.desc"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        total_pages = min(data.get('total_pages', 1), 10)  # Limit to 10 pages per studio
                        
                        for page in range(1, total_pages + 1):
                            if page > 1:  # Skip first page as we already fetched it
                                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={studio_id}&with_original_language=hi&page={page}&sort_by=popularity.desc"
                                response = requests.get(url)
                                if response.status_code != 200:
                                    continue
                                data = response.json()
                                
                            before_count = len(self.unique_movie_ids)
                            self._process_movie_results(data.get('results', []), is_bollywood=True)
                            after_count = len(self.unique_movie_ids)
                            bollywood_fetched += (after_count - before_count)
                            
                            self._save_progress(f"Bollywood studio {studio_id} (page {page})")
                            time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching from Bollywood studio {studio_id}: {e}")
        
        # 2.3 Fetch using search for popular Bollywood actors
        if bollywood_fetched < BOLLYWOOD_COUNT:
            bollywood_actors = [
                "Shah Rukh Khan", "Aamir Khan", "Salman Khan", "Amitabh Bachchan",
                "Akshay Kumar", "Hrithik Roshan", "Ranbir Kapoor", "Ranveer Singh",
                "Deepika Padukone", "Alia Bhatt", "Priyanka Chopra", "Katrina Kaif",
                "Kareena Kapoor", "Aishwarya Rai", "Madhuri Dixit", "Kajol",
                "Ajay Devgn", "Shahid Kapoor", "Varun Dhawan", "Sanjay Dutt"
            ]
            
            for actor in bollywood_actors:
                if bollywood_fetched >= BOLLYWOOD_COUNT:
                    break
                    
                # Search for actor and then filter their movies
                search_url = f"{TMDB_BASE_URL}/search/person?api_key={self.api_key}&query={actor.replace(' ', '+')}"
                try:
                    response = requests.get(search_url)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        if results:
                            person_id = results[0].get('id')
                            if person_id:
                                # Get movies the actor was in
                                credits_url = f"{TMDB_BASE_URL}/person/{person_id}/movie_credits?api_key={self.api_key}"
                                credits_response = requests.get(credits_url)
                                
                                if credits_response.status_code == 200:
                                    credits_data = credits_response.json()
                                    cast_movies = credits_data.get('cast', [])
                                    
                                    before_count = len(self.unique_movie_ids)
                                    # Process movies, favoring Hindi language
                                    for movie in cast_movies:
                                        movie_id = movie.get('id')
                                        if movie_id and movie_id not in self.unique_movie_ids:
                                            try:
                                                movie_details = self._get_movie_details(movie_id, prefer_hindi=True)
                                                if movie_details and movie_details.get('language') == 'hi':
                                                    # Add the 'bollywood' tag to make searching easier
                                                    movie_details['document'] += " bollywood hindi indian"
                                                    self.movies.append(movie_details)
                                                    self.unique_movie_ids.add(movie_id)
                                                    time.sleep(0.1)
                                            except Exception as e:
                                                print(f"Error processing Bollywood movie {movie_id}: {e}")
                                                
                                    after_count = len(self.unique_movie_ids)
                                    bollywood_fetched += (after_count - before_count)
                                    self._save_progress(f"Bollywood actor {actor}")
                except Exception as e:
                    print(f"Error fetching movies for Bollywood actor {actor}: {e}")
            
        # 3. Fetch South Indian movies (Tamil, Telugu, Malayalam, Kannada)
        print(f"Phase 3: Fetching South Indian movies (target: {SOUTH_INDIAN_COUNT})...")
        south_indian_fetched = 0
        south_indian_languages = {
            "ta": "Tamil", 
            "te": "Telugu", 
            "ml": "Malayalam", 
            "kn": "Kannada"
        }
                
        # 3.1 Fetch by South Indian languages
        for language_code, language_name in south_indian_languages.items():
            language_target = SOUTH_INDIAN_COUNT // len(south_indian_languages)
            language_count = 0
            page = 1
                    
            while language_count < language_target and south_indian_fetched < SOUTH_INDIAN_COUNT:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language={language_code}&page={page}&sort_by=popularity.desc"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        if not results:
                            break
                                
                        before_count = len(self.unique_movie_ids)
                        # Add tag for the language to make searching easier
                        for result in results:
                            result['language_tag'] = language_name.lower() + " south indian"
                            
                        self._process_movie_results(results, is_south_indian=True, language=language_name)
                        after_count = len(self.unique_movie_ids)
                        language_count += (after_count - before_count)
                        south_indian_fetched += (after_count - before_count)
                                
                        self._save_progress(f"{language_name} movies (page {page})")
                        page += 1
                        time.sleep(0.5)
                                
                        if page > data.get('total_pages', 1) or page > 20:  # Increased to 20 pages to get more results
                            break
                    else:
                        print(f"Error {response.status_code} when fetching {language_name} movies")
                        time.sleep(2)
                except Exception as e:
                    print(f"Exception while fetching {language_name} movies: {e}")
                    time.sleep(2)
                    
        # 3.2 Fetch by top South Indian actors/directors
        if south_indian_fetched < SOUTH_INDIAN_COUNT:
            south_indian_personalities = [
                # Tamil
                "Rajinikanth", "Vijay", "Ajith Kumar", "Kamal Haasan", "Suriya", 
                "Mani Ratnam", "AR Rahman", "Dhanush", "Vikram", "Nayanthara",
                # Telugu
                "Chiranjeevi", "Prabhas", "Allu Arjun", "Mahesh Babu", "Jr NTR",
                "Ram Charan", "SS Rajamouli", "Nagarjuna", "Nani", "Samantha Ruth Prabhu",
                # Malayalam
                "Mohanlal", "Mammootty", "Fahadh Faasil", "Dulquer Salmaan", "Nivin Pauly",
                # Kannada
                "Yash", "Sudeep", "Darshan", "Puneeth Rajkumar", "Upendra"
            ]
            
            for personality in south_indian_personalities:
                if south_indian_fetched >= SOUTH_INDIAN_COUNT:
                    break
                    
                # Search for person
                search_url = f"{TMDB_BASE_URL}/search/person?api_key={self.api_key}&query={personality.replace(' ', '+')}"
                try:
                    response = requests.get(search_url)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        if results:
                            person_id = results[0].get('id')
                            if person_id:
                                # Get movies the person was in
                                credits_url = f"{TMDB_BASE_URL}/person/{person_id}/movie_credits?api_key={self.api_key}"
                                credits_response = requests.get(credits_url)
                                
                                if credits_response.status_code == 200:
                                    credits_data = credits_response.json()
                                    # Get both acting and directing credits
                                    all_movies = credits_data.get('cast', []) + credits_data.get('crew', [])
                                    
                                    before_count = len(self.unique_movie_ids)
                                    # Process movies, but filter for South Indian languages
                                    for movie in all_movies:
                                        movie_id = movie.get('id')
                                        if movie_id and movie_id not in self.unique_movie_ids:
                                            try:
                                                movie_details = self._get_movie_details(movie_id)
                                                if movie_details and movie_details.get('language') in ["ta", "te", "ml", "kn"]:
                                                    # Add south indian tag for easier searching
                                                    movie_details['document'] += " south indian"
                                                    self.movies.append(movie_details)
                                                    self.unique_movie_ids.add(movie_id)
                                                    time.sleep(0.1)
                                            except Exception as e:
                                                print(f"Error processing South Indian movie {movie_id}: {e}")
                                                
                                    after_count = len(self.unique_movie_ids)
                                    south_indian_fetched += (after_count - before_count)
                                    self._save_progress(f"South Indian personality {personality}")
                except Exception as e:
                    print(f"Error fetching movies for South Indian personality {personality}: {e}")
        
        # 4. Fetch web series (TV shows) - both Hollywood and Bollywood
        print(f"Phase 4: Fetching web series (target: {WEB_SERIES_COUNT})...")
        web_series_fetched = 0
        
        # 4.1 Fetch popular Hollywood web series first (60% of web series target)
        hollywood_series_target = int(WEB_SERIES_COUNT * 0.6)
        hollywood_series_count = 0
        page = 1
        
        while hollywood_series_count < hollywood_series_target:
            url = f"{TMDB_BASE_URL}/tv/popular?api_key={self.api_key}&page={page}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                    
                    # Process TV shows similar to movies
                    before_count = len(self.unique_movie_ids)
                    for show in results:
                        if show.get('id') and show['id'] not in self.unique_movie_ids:
                            try:
                                show_details = self._get_tv_details(show['id'])
                                if show_details:
                                    # Add 'web series' tag for easier searching
                                    show_details['document'] += " web series tv show"
                                    self.movies.append(show_details)
                                    self.unique_movie_ids.add(show['id'])
                                    time.sleep(0.1)  # Prevent rate limiting
                            except Exception as e:
                                print(f"Error processing TV show {show.get('id')}: {e}")
                    
                    after_count = len(self.unique_movie_ids)
                    hollywood_series_count += (after_count - before_count)
                    web_series_fetched += (after_count - before_count)
                    
                    self._save_progress(f"Popular web series (page {page})")
                    page += 1
                    time.sleep(0.5)
                    
                    if page > data.get('total_pages', 1) or page > 15:  # Limit to 15 pages
                        break
                else:
                    print(f"Error {response.status_code} when fetching popular web series")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching popular web series: {e}")
                time.sleep(2)
        
        # 4.2 Fetch Hindi (Bollywood) web series specifically (40% of web series target)
        bollywood_series_target = WEB_SERIES_COUNT - hollywood_series_count
        bollywood_series_count = 0
        page = 1
        
        while bollywood_series_count < bollywood_series_target:
            url = f"{TMDB_BASE_URL}/discover/tv?api_key={self.api_key}&with_original_language=hi&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    if not results:
                        break
                    
                    before_count = len(self.unique_movie_ids)
                    for show in results:
                        if show.get('id') and show['id'] not in self.unique_movie_ids:
                            try:
                                show_details = self._get_tv_details(show['id'])
                                if show_details:
                                    # Add 'bollywood web series' tag for easier searching
                                    show_details['document'] += " bollywood hindi indian web series tv show"
                                    self.movies.append(show_details)
                                    self.unique_movie_ids.add(show['id'])
                                    time.sleep(0.1)
                            except Exception as e:
                                print(f"Error processing Hindi TV show {show.get('id')}: {e}")
                    
                    after_count = len(self.unique_movie_ids)
                    bollywood_series_count += (after_count - before_count)
                    web_series_fetched += (after_count - before_count)
                    
                    self._save_progress(f"Hindi web series (page {page})")
                    page += 1
                    time.sleep(0.5)
                    
                    if page > data.get('total_pages', 1) or page > 30:  # Increased to 30 pages to get more results
                        break
                else:
                    print(f"Error {response.status_code} when fetching Hindi web series")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching Hindi web series: {e}")
                time.sleep(2)
        
        # If we still need more Hindi web series, try OTT platforms
        if bollywood_series_count < bollywood_series_target:
            ott_platforms = [
                213, # Netflix
                2552, # Amazon
                3186, # HBO
                2360, # ZEE5
                2311, # Hotstar
                5420, # MX Player
                3584, # SonyLIV
                5322, # ALTBalaji
            ]
            
            for platform in ott_platforms:
                if bollywood_series_count >= bollywood_series_target:
                    break
                    
                url = f"{TMDB_BASE_URL}/discover/tv?api_key={self.api_key}&with_networks={platform}&with_original_language=hi&page=1"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        before_count = len(self.unique_movie_ids)
                        for show in results:
                            if show.get('id') and show['id'] not in self.unique_movie_ids:
                                try:
                                    show_details = self._get_tv_details(show['id'])
                                    if show_details:
                                        show_details['document'] += " bollywood hindi indian web series tv show"
                                        self.movies.append(show_details)
                                        self.unique_movie_ids.add(show['id'])
                                        time.sleep(0.1)
                                except Exception as e:
                                    print(f"Error processing OTT TV show {show.get('id')}: {e}")
                        
                        after_count = len(self.unique_movie_ids)
                        bollywood_series_count += (after_count - before_count)
                        web_series_fetched += (after_count - before_count)
                        
                        self._save_progress(f"OTT platform {platform}")
                except Exception as e:
                    print(f"Error fetching from OTT platform {platform}: {e}")
                
        print(f"Fetched {web_series_fetched} web series total")
        
        # Final save
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.movies, f, ensure_ascii=False, indent=2)
            
        elapsed_time = (time.time() - start_time) / 60
        
        # Print summary
        print("\n=== FINAL SUMMARY ===")
        print(f"Total movies/shows in dataset: {len(self.movies)}")
        print(f"- Hollywood/International: {hollywood_count}")
        print(f"- Bollywood: {bollywood_fetched}")
        print(f"- South Indian: {south_indian_fetched}")
        print(f"- Web Series: {web_series_fetched}")
        print(f"Fetched and saved in {elapsed_time:.2f} minutes")
    
    def _fetch_additional_data(self):
        """Fetch additional movies to reach the target count"""
        current_count = len(self.movies)
        if current_count >= TARGET_MOVIE_COUNT:
            return
            
        # How many more movies we need
        needed = TARGET_MOVIE_COUNT - current_count
        
        # Count movies by category in current dataset
        hollywood_count = 0
        bollywood_count = 0
        south_indian_count = 0
        web_series_count = 0
        
        for movie in self.movies:
            content_type = movie.get('content_type', 'movie')
            language = movie.get('language', 'unknown')
            
            if content_type == 'tv':
                web_series_count += 1
            elif language == 'hi':
                bollywood_count += 1
            elif language in ['ta', 'te', 'ml', 'kn']:
                south_indian_count += 1
            else:
                hollywood_count += 1
        
        print(f"\nCurrent counts:")
        print(f"- Hollywood/International: {hollywood_count}")
        print(f"- Bollywood: {bollywood_count}")
        print(f"- South Indian: {south_indian_count}")
        print(f"- Web Series: {web_series_count}")
        
        # Calculate how many more of each category we need
        need_hollywood = max(0, HOLLYWOOD_COUNT - hollywood_count)
        need_bollywood = max(0, BOLLYWOOD_COUNT - bollywood_count)
        need_south_indian = max(0, SOUTH_INDIAN_COUNT - south_indian_count)
        need_web_series = max(0, WEB_SERIES_COUNT - web_series_count)
        
        print(f"\nNeed to fetch:")
        print(f"- Hollywood/International: {need_hollywood}")
        print(f"- Bollywood: {need_bollywood}")
        print(f"- South Indian: {need_south_indian}")
        print(f"- Web Series: {need_web_series}")
        
        # First try to fetch more Bollywood movies if needed
        if need_bollywood > 0:
            print(f"Fetching {need_bollywood} more Bollywood movies...")
            # Using Hindi language filter
            self._fetch_by_language("hi", max_pages=min(50, need_bollywood // 20 + 1))
        
        # Then fetch more South Indian movies if needed
        if need_south_indian > 0:
            print(f"Fetching {need_south_indian} more South Indian movies...")
            languages = ["ta", "te", "ml", "kn"]
            for lang in languages:
                self._fetch_by_language(lang, max_pages=min(20, need_south_indian // 20 + 1))
        
        # Then fetch more web series if needed
        if need_web_series > 0:
            print(f"Fetching {need_web_series} more web series...")
            pages = min(20, need_web_series // 20 + 1)
            
            # Popular TV shows
            url = f"{TMDB_BASE_URL}/tv/popular?api_key={self.api_key}&page=1"
            for page in range(1, pages + 1):
            
                    response = requests.get(url.replace("page=1", f"page={page}"))
                    if response.status_code == 200:
                        data = response.json()
                        for show in data.get('results', []):
                            if show.get('id') and show['id'] not in self.unique_movie_ids:
                                try:
                                    show_details = self._get_tv_details(show['id'])
                                    if show_details:
                                        show_details['document'] += " web series tv show"
                                        self.movies.append(show_details)
                                        self.unique_movie_ids.add(show['id'])
                                        time.sleep(0.1)
                                except Exception as e:
                                    print(f"Error processing TV show {show.get('id')}: {e}")
        
        # Finally, fetch more Hollywood movies if needed
        if need_hollywood > 0:
            print(f"Fetching {need_hollywood} more Hollywood/International movies...")
            
            # Try to fetch by years we might not have covered yet
            years_to_try = list(range(1980, datetime.now().year))
            random.shuffle(years_to_try)  # Randomize to get variety
            
            for year in years_to_try:
                if len(self.unique_movie_ids) >= TARGET_MOVIE_COUNT:
                    break
                self._fetch_by_year(year, max_pages=2, strict_year=True)
        
        # Save final dataset
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.movies, f, ensure_ascii=False, indent=2)
            
        print(f"Dataset updated to {len(self.movies)} movies/shows")
    
    def _get_genres(self):
        """Get list of all available movie genres from TMDB"""
        url = f"{TMDB_BASE_URL}/genre/movie/list?api_key={self.api_key}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json().get('genres', [])
            else:
                print(f"Error {response.status_code} when fetching genres")
                return []
        except Exception as e:
            print(f"Exception while fetching genres: {e}")
            return []
    
    def _fetch_from_endpoint(self, endpoint, pages=10):
        """Fetch movies from a specific TMDB endpoint"""
        for page in range(1, pages + 1):
            url = f"{TMDB_BASE_URL}/{endpoint}?api_key={self.api_key}&page={page}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched {endpoint} page {page}/{pages}")
                    time.sleep(0.5)  # Prevent rate limiting
                else:
                    print(f"Error {response.status_code} when fetching {endpoint} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching {endpoint} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_year(self, year, max_pages=5, strict_year=False):
        """Fetch movies released in a specific year"""
        for page in range(1, max_pages + 1):
            # For strict year matching, use both primary_release_year and year parameters
            if strict_year:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&primary_release_year={year}&year={year}&page={page}&sort_by=popularity.desc"
            else:
                url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&primary_release_year={year}&page={page}&sort_by=popularity.desc"
            
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    # Additional verification for strict year matching
                    if strict_year:
                        filtered_results = []
                        for movie in results:
                            release_date = movie.get('release_date', '')
                            if release_date and release_date.startswith(str(year)):
                                filtered_results.append(movie)
                        results = filtered_results
                    
                    self._process_movie_results(results)
                    print(f"Fetched year {year} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching year {year} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching year {year} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_genre(self, genre_id, genre_name, max_pages=5):
        """Fetch movies by genre"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_genres={genre_id}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    # Add genre tag for easier searching
                    for movie in results:
                        movie['genre_tag'] = genre_name.lower()
                        
                    self._process_movie_results(results)
                    print(f"Fetched genre {genre_name} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching genre {genre_name} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching genre {genre_name} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_language(self, language_code, max_pages=10):
        """Fetch movies by original language"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_original_language={language_code}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched language {language_code} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching language {language_code} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching language {language_code} page {page}: {e}")
                time.sleep(2)
    
    def _fetch_by_company(self, company_id, max_pages=5):
        """Fetch movies by production company"""
        for page in range(1, max_pages + 1):
            url = f"{TMDB_BASE_URL}/discover/movie?api_key={self.api_key}&with_companies={company_id}&page={page}&sort_by=popularity.desc"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    self._process_movie_results(data.get('results', []))
                    print(f"Fetched company {company_id} page {page}/{max_pages}")
                    time.sleep(0.5)
                else:
                    print(f"Error {response.status_code} when fetching company {company_id} page {page}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception while fetching company {company_id} page {page}: {e}")
                time.sleep(2)
    
    def _process_movie_results(self, results, is_bollywood=False, is_south_indian=False, language=None):
        """Process movie results and add to dataset if not already present"""
        count = 0
        for movie in results:
            movie_id = movie.get('id')
            if movie_id and movie_id not in self.unique_movie_ids:
                try:
                    # Get additional movie details
                    movie_details = self._get_movie_details(movie_id)
                    
                    if movie_details:
                        # Add specific tags based on movie type
                        if is_bollywood:
                            movie_details['document'] += " bollywood hindi indian"
                        elif is_south_indian:
                            movie_details['document'] += f" {language.lower()} south indian"
                        
                        # Add any genre or language tags that were added during fetching
                        if 'genre_tag' in movie:
                            movie_details['document'] += f" {movie['genre_tag']}"
                        if 'language_tag' in movie:
                            movie_details['document'] += f" {movie['language_tag']}"
                        
                        self.movies.append(movie_details)
                        self.unique_movie_ids.add(movie_id)
                        count += 1
                except Exception as e:
                    print(f"Error processing movie {movie_id}: {e}")
        
        print(f"Added {count} new movies from batch")
    
    def _get_movie_details(self, movie_id, prefer_hindi=False):
        """Get detailed information about a specific movie"""
        url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={self.api_key}&append_to_response=credits,keywords"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Basic movie information
                title = data.get('title', '')
                original_title = data.get('original_title', '')
                overview = data.get('overview', '')
                release_date = data.get('release_date', '')
                
                # Language handling - for Bollywood preferences
                original_language = data.get('original_language', '')
                if prefer_hindi and original_language != 'hi':
                    return None
                
                # Get genres, cast, crew
                genres = [genre['name'] for genre in data.get('genres', [])]
                
                # Get director and top cast
                director = ""
                cast = []
                
                credits = data.get('credits', {})
                crew = credits.get('crew', [])
                actors = credits.get('cast', [])
                
                for person in crew:
                    if person.get('job') == 'Director':
                        director = person.get('name', '')
                        break
                
                for actor in actors[:10]:  # Get top 10 cast
                    if actor.get('name'):
                        cast.append(actor.get('name'))
                
                # Get keywords/tags
                keywords = []
                if 'keywords' in data and 'keywords' in data['keywords']:
                    keywords = [kw['name'] for kw in data['keywords']['keywords']]
                
                # Create a comprehensive document for text search
                document = f"{title} {original_title} {overview} "
                document += f"{' '.join(genres)} {director} {' '.join(cast)} {' '.join(keywords)} "
                document += f"{release_date[:4] if release_date else ''} "  # Add year for searching by year
                
                # Add movie or specific category identifiers
                document += "movie film "
                
                # Add language specific identifiers
                if original_language == 'en':
                    document += "english hollywood international "
                elif original_language == 'hi':
                    document += "hindi bollywood indian "
                elif original_language == 'ta':
                    document += "tamil south indian "
                elif original_language == 'te':
                    document += "telugu south indian "
                elif original_language == 'ml':
                    document += "malayalam south indian "
                elif original_language == 'kn':
                    document += "kannada south indian "
                
                # Store the movie data
                movie_data = {
                    'id': movie_id,
                    'title': title,
                    'original_title': original_title,
                    'overview': overview,
                    'poster_path': data.get('poster_path', ''),
                    'release_date': release_date,
                    'genres': genres,
                    'director': director,
                    'cast': cast,
                    'language': original_language,
                    'document': document,
                    'content_type': 'movie',
                    'keywords': keywords
                }
                
                return movie_data
            else:
                print(f"Error {response.status_code} when fetching movie {movie_id}")
                return None
        except Exception as e:
            print(f"Exception while fetching movie {movie_id}: {e}")
            return None
    
    def _get_tv_details(self, show_id):
        """Get detailed information about a TV show"""
        url = f"{TMDB_BASE_URL}/tv/{show_id}?api_key={self.api_key}&append_to_response=credits,keywords"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # Basic show information
                title = data.get('name', '')
                original_title = data.get('original_name', '')
                overview = data.get('overview', '')
                first_air_date = data.get('first_air_date', '')
                
                # Get genres, cast, crew
                genres = [genre['name'] for genre in data.get('genres', [])]
                
                # Get creator and top cast
                creators = []
                cast = []
                
                creator_data = data.get('created_by', [])
                for creator in creator_data:
                    if creator.get('name'):
                        creators.append(creator.get('name'))
                
                credits = data.get('credits', {})
                actors = credits.get('cast', [])
                
                for actor in actors[:10]:  # Get top 10 cast
                    if actor.get('name'):
                        cast.append(actor.get('name'))
                
                # Get keywords/tags
                keywords = []
                if 'keywords' in data and 'results' in data['keywords']:
                    keywords = [kw['name'] for kw in data['keywords']['results']]
                
                # Create a comprehensive document for text search
                document = f"{title} {original_title} {overview} "
                document += f"{' '.join(genres)} {' '.join(creators)} {' '.join(cast)} {' '.join(keywords)} "
                document += f"{first_air_date[:4] if first_air_date else ''} "  # Add year
                
                # Add TV identifiers
                document += "tv television series show web series streaming "
                
                # Add language specific identifiers
                original_language = data.get('original_language', '')
                if original_language == 'en':
                    document += "english hollywood international "
                elif original_language == 'hi':
                    document += "hindi bollywood indian "
                elif original_language == 'ta':
                    document += "tamil south indian "
                elif original_language == 'te':
                    document += "telugu south indian "
                elif original_language == 'ml':
                    document += "malayalam south indian "
                elif original_language == 'kn':
                    document += "kannada south indian "
                
                # Add streaming platform information if available
                networks = data.get('networks', [])
                network_names = []
                for network in networks:
                    name = network.get('name', '').lower()
                    network_names.append(name)
                    document += f"{name} "
                    
                    # Add common OTT platform keywords
                    if 'netflix' in name:
                        document += "netflix ott streaming "
                    elif 'amazon' in name or 'prime' in name:
                        document += "amazon prime video ott streaming "
                    elif 'disney' in name or 'hotstar' in name:
                        document += "disney+ hotstar ott streaming "
                    elif 'hbo' in name:
                        document += "hbo max ott streaming "
                    elif 'hulu' in name:
                        document += "hulu ott streaming "
                    elif 'zee' in name:
                        document += "zee5 ott streaming "
                    elif 'sony' in name:
                        document += "sonyliv ott streaming "
                    elif 'alt' in name or 'balaji' in name:
                        document += "altbalaji ott streaming "
                
                # Store the TV show data
                show_data = {
                    'id': show_id,
                    'title': title,
                    'original_title': original_title,
                    'overview': overview,
                    'poster_path': data.get('poster_path', ''),
                    'first_air_date': first_air_date,
                    'genres': genres,
                    'creators': creators,
                    'cast': cast,
                    'language': original_language,
                    'document': document,
                    'content_type': 'tv',
                    'keywords': keywords,
                    'networks': network_names
                }
                
                return show_data
            else:
                print(f"Error {response.status_code} when fetching TV show {show_id}")
                return None
        except Exception as e:
            print(f"Exception while fetching TV show {show_id}: {e}")
            return None
    
    def _save_progress(self, stage_name):
        """Save progress after completing a stage of data fetching"""
        print(f"Progress update: {stage_name} - Total movies/shows: {len(self.movies)}")
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.movies, f, ensure_ascii=False)
    
    def _prepare_tfidf(self):
        """Prepare TF-IDF matrix for movie recommendations"""
        print("Preparing TF-IDF matrix for recommendations...")
        
        # Extract document text
        documents = [self._preprocess_text(movie['document']) for movie in self.movies]
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def _preprocess_text(self, text):
        """Preprocess text for better NLP performance"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def recommend_movies(self, query, n=10, filters=None):
        """Recommend movies based on query and optional filters"""
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Convert query to TF-IDF vector
        query_vec = self.vectorizer.transform([processed_query])
        
        # Calculate similarity scores
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Apply filters if specified
        if filters:
            filtered_indices = self._apply_filters(filters)
            
            # If we have valid filtered indices
            if filtered_indices:
                # Zero out similarity scores for non-matching indices
                mask = np.ones(len(cosine_similarities), dtype=bool)
                mask[filtered_indices] = False
                cosine_similarities[mask] = 0
        
        # Get top N similar movies
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        
        # Create recommendations list
        recommendations = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:  # Only include movies with non-zero similarity
                movie = self.movies[idx]
                recommendations.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'poster_path': movie['poster_path'],
                    'release_date': movie.get('release_date', movie.get('first_air_date', '')),
                    'similarity_score': float(cosine_similarities[idx]),
                    'content_type': movie.get('content_type', 'movie')
                })
        
        return recommendations
    
    def _apply_filters(self, filters):
        """Apply filters to narrow down movie recommendations"""
        filtered_indices = []
        
        for i, movie in enumerate(self.movies):
            include = True
            
            # Year filter
            if 'year' in filters:
                year_str = ''
                if movie.get('release_date'):
                    year_str = movie['release_date'][:4]
                elif movie.get('first_air_date'):
                    year_str = movie['first_air_date'][:4]
                
                if year_str != str(filters['year']):
                    include = False
            
            # Genre filter
            if 'genre' in filters and include:
                if not any(genre.lower() == filters['genre'].lower() for genre in movie.get('genres', [])):
                    include = False
            
            # Language filter
            if 'language' in filters and include:
                if movie.get('language') != filters['language']:
                    include = False
            
            # Content type filter
            if 'content_type' in filters and include:
                if movie.get('content_type') != filters['content_type']:
                    include = False
            
            # Time period filter
            if 'time_period' in filters and include:
                start_year, end_year = filters['time_period']
                year = 0
                if movie.get('release_date'):
                    try:
                        year = int(movie['release_date'][:4])
                    except ValueError:
                        pass
                elif movie.get('first_air_date'):
                    try:
                        year = int(movie['first_air_date'][:4])
                    except ValueError:
                        pass
                
                if not (start_year <= year <= end_year):
                    include = False
            
            # Add more filters as needed
            
            if include:
                filtered_indices.append(i)
        
        return filtered_indices
    
    def extract_keywords_from_query(self, query):
        """Extract keywords from query for better understanding user intent"""
        # Common keywords to look for
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        genre_keywords = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller", 
                         "adventure", "fantasy", "animation", "documentary", "biography"]
        region_keywords = {
            "hollywood": "en",
            "bollywood": "hi",
            "hindi": "hi",
            "tamil": "ta",
            "telugu": "te",
            "malayalam": "ml", 
            "kannada": "kn",
            "south indian": ["ta", "te", "ml", "kn"],
            "indian": ["hi", "ta", "te", "ml", "kn"]
        }
        content_type_keywords = {
            "movie": "movie",
            "film": "movie",
            "web series": "tv",
            "tv series": "tv",
            "show": "tv",
            "television": "tv",
            "ott": "tv",
            "streaming": "tv"
        }
        
        # Extract filters
        filters = {}
        
        # Extract year
        year_match = re.search(year_pattern, query)
        if year_match:
            filters['year'] = int(year_match.group())
        
        # Extract genre
        for genre in genre_keywords:
            if re.search(r'\b' + genre + r'\b', query.lower()):
                filters['genre'] = genre
                break
        
        # Extract region/language
        for region, language in region_keywords.items():
            if re.search(r'\b' + region + r'\b', query.lower()):
                filters['language'] = language
                break
        
        # Extract content type
        for content_type, value in content_type_keywords.items():
            if re.search(r'\b' + content_type + r'\b', query.lower()):
                filters['content_type'] = value
                break
        
        # Extract feeling/mood
        feeling_mapping = {
            # Map feelings to genres/keywords
            "happy": ["comedy", "feel-good", "uplifting"],
            "sad": ["drama", "tragedy", "emotional"],
            "scary": ["horror", "thriller", "suspense"],
            "exciting": ["action", "adventure", "thriller"],
            "thoughtful": ["drama", "philosophical", "thought-provoking"],
            "romantic": ["romance", "love story", "romantic comedy"]
        }
        
        for feeling, keywords in feeling_mapping.items():
            if re.search(r'\b' + feeling + r'\b', query.lower()):
                # Add these keywords to the query itself rather than as filters
                query += ' ' + ' '.join(keywords)
                break
        
        return query, filters

# Create API endpoints
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query', '')
    n = int(data.get('n', 10))
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Extract keywords from query
    enhanced_query, filters = recommender.extract_keywords_from_query(query)
    
    # Override filters if explicitly provided
    for filter_key in ['year', 'genre', 'language', 'content_type']:
        if filter_key in data:
            filters[filter_key] = data[filter_key]
    
    # Get recommendations
    recommendations = recommender.recommend_movies(enhanced_query, n, filters)
    
    return jsonify({
        'original_query': query,
        'enhanced_query': enhanced_query,
        'filters_applied': filters,
        'recommendations': recommendations
    })

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    for movie in recommender.movies:
        if movie['id'] == movie_id:
            return jsonify(movie)
    
    return jsonify({'error': 'Movie not found'}), 404

@app.route('/api/filters', methods=['GET'])
def get_filters():
    # Extract unique values for filters
    languages = set()
    genres = set()
    years = set()
    content_types = set()
    
    for movie in recommender.movies:
        if 'language' in movie:
            languages.add(movie['language'])
        
        for genre in movie.get('genres', []):
            genres.add(genre)
        
        # Extract year from release date or first air date
        year = None
        if 'release_date' in movie and movie['release_date']:
            try:
                year = int(movie['release_date'][:4])
            except ValueError:
                pass
        elif 'first_air_date' in movie and movie['first_air_date']:
            try:
                year = int(movie['first_air_date'][:4])
            except ValueError:
                pass
                
        if year:
            years.add(year)
            
        if 'content_type' in movie:
            content_types.add(movie['content_type'])
    
    # Map language codes to names
    language_names = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'ml': 'Malayalam',
        'kn': 'Kannada'
    }
    
    languages_with_names = [{'code': code, 'name': language_names.get(code, code)} for code in languages]
    
    return jsonify({
        'languages': sorted(languages_with_names, key=lambda x: x['name']),
        'genres': sorted(list(genres)),
        'years': sorted(list(years)),
        'content_types': sorted(list(content_types))
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Calculate statistics about the dataset
    total_count = len(recommender.movies)
    
    # Count by content type
    movies_count = sum(1 for movie in recommender.movies if movie.get('content_type') == 'movie')
    tv_count = sum(1 for movie in recommender.movies if movie.get('content_type') == 'tv')
    
    # Count by language/region
    hollywood_count = sum(1 for movie in recommender.movies if movie.get('content_type') == 'movie' and movie.get('language') == 'en')
    bollywood_count = sum(1 for movie in recommender.movies if movie.get('language') == 'hi')
    south_indian_count = sum(1 for movie in recommender.movies if movie.get('language') in ['ta', 'te', 'ml', 'kn'])
    
    # Genre distribution
    genre_counts = {}
    for movie in recommender.movies:
        for genre in movie.get('genres', []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Year distribution
    year_counts = {}
    for movie in recommender.movies:
        year = None
        if movie.get('release_date'):
            try:
                year = int(movie['release_date'][:4])
            except ValueError:
                pass
        elif movie.get('first_air_date'):
            try:
                year = int(movie['first_air_date'][:4])
            except ValueError:
                pass
        
        if year:
            year_counts[year] = year_counts.get(year, 0) + 1
    
    return jsonify({
        'total_count': total_count,
        'content_type_distribution': {
            'movies': movies_count,
            'tv_shows': tv_count
        },
        'region_distribution': {
            'hollywood': hollywood_count,
            'bollywood': bollywood_count,
            'south_indian': south_indian_count
        },
        'top_genres': dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]),
        'year_distribution': dict(sorted(year_counts.items()))
    })

# Initialize recommender when script runs
print("Initializing Movie Recommender...")
recommender = MovieRecommender()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)