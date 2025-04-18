🛍️ Multi-Modal Product Recommendation System
This project is a Flask-based web application that provides a product recommendation system using both text and image inputs. It leverages NLP and Computer Vision models to deliver smart suggestions from a fashion dataset.

🚀 Features
🔍 Text-Based Recommendations: Enter product-related keywords or descriptions and get relevant items.
🖼️ Image-Based Recommendations: Upload a product image and discover visually similar products.
🤖 Deep Learning Models:
SentenceTransformer (MiniLM) for text embeddings
ResNet50 for image feature extraction
📊 Evaluation Metrics: Precision, Recall, and F1-score for recommendation evaluation.

🧰 Technologies Used
Python
Flask
Pandas, NumPy
Keras (with TensorFlow backend)
SentenceTransformers (paraphrase-MiniLM-L6-v2)
Scikit-learn
PIL (Pillow)

📁 Folder Structure
project/
│
├── static/
│   └── images/              # Product images
│
├── templates/
│   ├── index.html           # Homepage UI
│   └── results.html         # Results page
│
├── data/
│   ├── products.csv         # Text-based product data
│   └── styles.csv           # Image-style data
│
├── app.py                   # Main Flask application
└── README.md                # Project documentation


⚙️ Setup Instructions
1. Clone the repository
   git clone https://github.com/ChAtulKumarPrusty/Product-Recommendation-System.git
cd product-recommendation-system

2. Install dependencies
pip install -r requirements.txt

3. Download/prep data
Ensure data/products.csv and data/styles.csv exist.
Place images in the static/images/ folder and name them as {id}.jpg.

4. Run the app
python app.py
Access it at: http://127.0.0.1:5000

📷 Example Usage
Text Input
"Blue floral dress"
👉 Returns top 5 products with similar titles/descriptions.
Image Input
Upload an image of a shirt
👉 Returns visually similar product images using ResNet50 embeddings.

🧪 Evaluation
The system supports evaluation using precision, recall, and F1-score based on the relevance of recommended items.

✨ Future Improvements
Add support for hybrid recommendations (text + image)

UI enhancements with filters (e.g., category, price)

Deploy on cloud (Heroku, AWS)

🙌 Authors
Ch Atul Kumar Prusty

Let me know if you want a requirements.txt or a badge-enhanced version of this README!
