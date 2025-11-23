const STATIC_EXPORTS = [
    "visualizations/analysis_report.txt",
    "visualizations/correlation_matrix.png",
    "visualizations/daily_patterns.png",
    "visualizations/engagement_trends.png",
    "visualizations/hourly_engagement.png",
    "visualizations/platform_comparison.png",
    "visualizations/sentiment_analysis.png",
    "visualizations/top_posts_table.png",
    "visualizations/topic_distribution.png",
    "visualizations/wordcount_engagement.png"
];

document.addEventListener("DOMContentLoaded", () => {
    renderStaticAssets();
});

function renderStaticAssets() {
    const container = document.getElementById("static-assets");
    container.innerHTML = ""; // clear old content

    STATIC_EXPORTS.forEach(file => {
        const card = document.createElement("div");
        card.className = "asset-card";

        const title = document.createElement("h3");
        title.textContent = file;

        const download = document.createElement("a");
        download.href = file;
        download.download = file;
        download.textContent = "⬇ Download";
        download.className = "download-btn";

        card.appendChild(title);

        if (file.endsWith(".png")) {
            const img = document.createElement("img");
            img.src = file;
            img.className = "asset-image";
            card.appendChild(img);
        }

        if (file.endsWith(".txt")) {
            const txt = document.createElement("iframe");
            txt.src = file;
            txt.className = "asset-text";
            card.appendChild(txt);
        }

        card.appendChild(download);
        container.appendChild(card);
    });
}
