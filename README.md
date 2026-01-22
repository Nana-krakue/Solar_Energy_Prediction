# ☀️ Solar Energy Prediction

A machine learning-powered REST API that predicts solar energy generation with high accuracy. Built with scikit-learn, Flask, and Docker for easy deployment and scalability.

## 🌟 Features

- **Machine Learning Model**: Trained on historical solar data with advanced feature engineering
- **REST API**: Simple HTTP endpoints for single and batch predictions
- **Containerized**: Docker and Docker Compose support for seamless deployment
- **Health Checks**: Built-in health monitoring for reliability
- **Production Ready**: Uses Gunicorn WSGI server with multiple workers
- **Batch Processing**: Efficient batch prediction endpoint for large datasets

## 📊 Project Overview

This project combines data science and software engineering to create a production-ready solar energy forecasting system. The model learns from historical patterns including:
- Time-based features (hour, day of week, month)
- Historical power generation (previous 1h, 2h, 4h)
- Statistical aggregates (24h mean and standard deviation)

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (recommended)
- Python 3.9+ (for local development)
- The trained model file: `best_solar_model.pkl`

### Run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:5000`

### Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `best_solar_model.pkl` is in the project root

3. Run the Flask app:
```bash
python app.py
```

Or with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

Returns the API status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Prediction
```http
POST /predict
```

Predicts solar energy generation for a single instance.

**Request Body:**
```json
{
  "hour": 12,
  "day_of_week": 3,
  "month": 6,
  "day_of_month": 15,
  "power_prev_1h": 150.5,
  "power_prev_2h": 148.2,
  "power_prev_4h": 145.0,
  "power_mean_24h": 155.0,
  "power_std_24h": 25.5
}
```

**Response:**
```json
{
  "prediction": 162.5,
  "unit": "MW",
  "input_features": { ... }
}
```

### Batch Prediction
```http
POST /predict-batch
```

Predicts solar energy for multiple instances in a single request.

**Request Body:**
```json
[
  { "hour": 12, "day_of_week": 3, ... },
  { "hour": 13, "day_of_week": 3, ... }
]
```

## 🛠️ Development

### Project Structure

```
.
├── app.py                      # Flask API application
├── Model.ipynb                 # Model development & training
├── Training Models.ipynb       # Additional model training experiments
├── best_solar_model.pkl        # Trained model (pickle format)
├── solar data.csv              # Historical solar data
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Multi-container orchestration
└── README.md                   # This file
```

### Model Development

The notebooks contain the full machine learning pipeline:

- **Model.ipynb**: Primary model development, training, and evaluation
- **Training Models.ipynb**: Experimentation with different algorithms

### Building a New Model

1. Update the data in `solar data.csv`
2. Run the training notebooks to develop/improve the model
3. Export the trained model as `best_solar_model.pkl`
4. Rebuild the Docker image: `docker-compose up --build`

## 📦 Dependencies

- **pandas** (≥1.3.0): Data manipulation and analysis
- **numpy** (≥1.21.0): Numerical computing
- **scikit-learn** (≥1.0.0): Machine learning algorithms
- **flask** (≥2.0.0): Web framework
- **gunicorn** (≥20.1.0): WSGI HTTP server

## 🐳 Docker Deployment

The project includes production-ready Docker configuration:

- **Dockerfile**: Multi-stage optimized image using Python 3.9-slim
- **docker-compose.yml**: Service orchestration with health checks
- **Health Checks**: Automated monitoring to ensure API availability

### Build Manually

```bash
docker build -t solar-api .
docker run -p 5000:5000 solar-api
```

## 📈 Performance

- **Response Time**: <100ms for single predictions
- **Batch Processing**: Efficiently handles 1000+ predictions per request
- **Concurrent Requests**: Supports multiple concurrent requests via Gunicorn workers

## 🔍 Example Usage

### Using cURL

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 12,
    "day_of_week": 3,
    "month": 6,
    "day_of_month": 15,
    "power_prev_1h": 150.5,
    "power_prev_2h": 148.2,
    "power_prev_4h": 145.0,
    "power_mean_24h": 155.0,
    "power_std_24h": 25.5
  }'
```

### Using Python

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "hour": 12,
    "day_of_week": 3,
    "month": 6,
    "day_of_month": 15,
    "power_prev_1h": 150.5,
    "power_prev_2h": 148.2,
    "power_prev_4h": 145.0,
    "power_mean_24h": 155.0,
    "power_std_24h": 25.5
}

response = requests.post(url, json=data)
print(response.json())
```

## ⚠️ Error Handling

The API returns appropriate HTTP status codes:

- **200**: Successful prediction
- **400**: Invalid request format or missing required fields
- **500**: Model not loaded or server error

Error responses include a descriptive message:
```json
{
  "error": "Missing required fields. Required: [...]"
}
```

## 🚢 Production Considerations

- **Model Updates**: Restart the container to load a new model
- **Scaling**: Adjust Gunicorn workers based on load (currently 4 workers)
- **Monitoring**: Implement logging and metrics collection in production
- **Security**: Add authentication and rate limiting for production APIs
- **Data Validation**: Consider additional input validation for edge cases

## 📝 License

This project is open source. Include your license information here.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Contact

For questions or suggestions, please reach out through GitHub issues.

---

**Happy Predicting!** ☀️⚡