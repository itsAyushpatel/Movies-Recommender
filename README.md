# 🎬 Movie Magic Recommender

A powerful movie recommendation system that helps users discover their next favorite movies and TV shows through natural language queries. Built with Python and machine learning, featuring an intuitive web interface.

![Image Preview](https://github.com/user-attachments/assets/c60a6401-9c1a-4bdd-b1e6-db02bedc16bf)

## 🌟 About The Project

**Movie Magic Recommender** is an intelligent movie recommendation engine created by **Ayush Patel**. The system features a comprehensive database of over 4,700 movies and TV shows, including Hollywood blockbusters, Bollywood films, South Indian cinema, and popular web series. Using advanced natural language processing and TF-IDF vectorization, it provides personalized recommendations based on user queries.

### Key Features
- 🔍 Natural language search capability
- 🎯 Smart filtering by region (Hollywood, Bollywood, South Indian)
- 📅 Year-based filtering
- 🎭 Genre-based recommendations
- 📺 Support for both movies and TV shows/web series
- 🌐 Modern, responsive web interface
- 💪 Real-time recommendations with similarity scoring

## 🛠️ Technologies Used

### Backend
- Python 3.x
- Flask (Web Framework)
- NLTK (Natural Language Processing)
- Scikit-learn (TF-IDF Vectorization & Cosine Similarity)
- TMDB API (Movie Database)
- NumPy (Numerical operations)
- Flask-CORS (Cross-Origin Resource Sharing)

### Frontend
- HTML5
- CSS3
- JavaScript (ES6+)
- Responsive Design

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-magic-recommender.git
   cd movie-magic-recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask flask-cors nltk scikit-learn numpy requests
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('stopwords')
   ```

5. **Add your TMDB API key**
   - Create a `tmdb_api_key.txt` file in the project root
   - Add your TMDB API key to this file

6. **Run the application**
   ```bash
   python movie_recommender.py
   ```

7. **Open the web interface**
   - Open `index.html` in your browser
   - Make sure the Flask server is running

## 💻 Usage

1. **Search for movies:** Enter a natural language query in the search bar:
   - "romantic bollywood movie 2015"
   - "horror web series netflix"
   - "tamil action films recent"
   - "sci-fi hollywood 2020"

2. **View recommendations:** The system will display relevant movies/shows with:
   - Movie posters
   - Title and release date
   - Similarity score (how well it matches your query)

## 📊 Dataset

The application maintains a database of:
- 2,500 Hollywood/International movies
- 1,000 Bollywood movies
- 500 South Indian movies
- 500 Web series (both Hollywood and Bollywood)

Data is fetched from TMDB API and includes:
- Title, overview, and release date
- Cast and crew information
- Genre and keywords
- Original language and region
- Poster images

## 🧠 How It Works

1. **Query Processing:** User queries are processed using NLTK for tokenization, stop word removal, and lemmatization.
2. **Feature Extraction:** TF-IDF vectorization converts processed text into numerical features.
3. **Similarity Calculation:** Cosine similarity compares the query vector with movie feature vectors.
4. **Filtering:** Optional filters for genre, year, language, and content type refine results.
5. **Ranking:** Movies are ranked by similarity score and displayed to the user.

## 🔑 API Endpoints

- `POST /api/recommend` - Get movie recommendations based on query
- `GET /api/movie/{id}` - Get detailed information about a specific movie
- `GET /api/filters` - Get available filter options
- `GET /api/stats` - Get statistics about the movie database

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
**Ayush Patel**

## 🙏 Acknowledgments

- TMDB API for providing comprehensive movie data
- scikit-learn for machine learning tools
- NLTK for natural language processing capabilities
- Flask for the lightweight web framework
