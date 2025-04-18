import os
import io
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Load data for text-based recommendation
product_data = pd.read_csv('data/products.csv')

# Load style data for image-based recommendation
style_data = pd.read_csv('data/styles.csv', on_bad_lines='skip')
image_folder = 'static/images/'
style_data['image_path'] = style_data['id'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

# Load model for text embedding
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize the ResNet50 model
image_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Precompute image embeddings for each product in style_data
def compute_style_embeddings():
    embeddings = []
    for img_path in style_data['image_path']:
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            embedding = image_model.predict(img_array).flatten()
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            embeddings.append(np.zeros((2048,)))
    style_data['embedding'] = embeddings

compute_style_embeddings()

def find_similar_text_products(query):
    query_embedding = text_model.encode([query])
    title_embeddings = text_model.encode(product_data['title'].tolist())
    similarities = cosine_similarity(query_embedding, title_embeddings).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    return product_data.iloc[top_indices]

def find_similar_image_products(uploaded_image):
    try:
        img = Image.open(io.BytesIO(uploaded_image.read()))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        uploaded_embedding = image_model.predict(img_array).flatten()

        similarities = cosine_similarity([uploaded_embedding], np.stack(style_data['embedding'].values)).flatten()
        top_index = similarities.argmax()
        matched_id = style_data.iloc[top_index]['id']

        return style_data[style_data['id'] == matched_id].drop(columns=['embedding'])
    except Exception as e:
        print("Error processing image:", e)
        return style_data.sample(5)

def evaluate_recommendation(y_true, recommended_indices):
    y_pred = [1 if i in recommended_indices else 0 for i in range(len(y_true))]
    if any(y_true) and any(y_pred):
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_type = request.form['input_type']
        if input_type == 'text':
            query = request.form['query']
            results = find_similar_text_products(query)
            return render_template('results.html', products=results.to_dict(orient='records'))
        elif input_type == 'image':
            image_file = request.files['image_file']
            if image_file:
                results = find_similar_image_products(image_file)
                return render_template('results.html', products=results.to_dict(orient='records'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
