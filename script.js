document.addEventListener("DOMContentLoaded", () => {
    const searchBtn = document.getElementById("search-button");
    const searchInput = document.getElementById("search-input");
    const resultsGrid = document.getElementById("results-grid");
    const resultsCount = document.getElementById("results-count");
    const resultsHeader = document.getElementById("results-header");
    const noResults = document.getElementById("no-results");
    const loading = document.getElementById("loading");
  
    searchBtn.addEventListener("click", async () => {
      const query = searchInput.value.trim();
      if (!query) return;
  
      // UI feedback
      resultsGrid.innerHTML = "";
      resultsHeader.classList.add("hidden");
      noResults.classList.add("hidden");
      loading.classList.remove("hidden");
  
      try {
        const response = await fetch("http://localhost:5000/api/recommend", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ query })
        });
  
        const data = await response.json();
        loading.classList.add("hidden");
  
        const recommendations = data.recommendations || [];
  
        if (recommendations.length > 0) {
          resultsCount.textContent = recommendations.length;
          resultsHeader.classList.remove("hidden");
  
          recommendations.forEach(movie => {
            const card = document.createElement("div");
            card.className = "movie-card";
            card.innerHTML = `
              <img src="${movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : 'https://via.placeholder.com/300x450?text=No+Image'}" alt="${movie.title}" />
              <div class="card-info">
                <div class="card-title">${movie.title}</div>
                <div class="card-meta">${movie.release_date || "N/A"}</div>
                <div class="similarity-score">⭐ ${movie.similarity_score.toFixed(2)}</div>
              </div>
            `;
            resultsGrid.appendChild(card);
          });
        } else {
          noResults.classList.remove("hidden");
        }
      } catch (error) {
        console.error("Error:", error);
        loading.classList.add("hidden");
        noResults.classList.remove("hidden");
      }
    });
  });
  