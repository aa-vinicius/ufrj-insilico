MOCK_MODEL_RESULTS = [
    {
        "id": 1, 
        "model": "rain_prediction", 
        "result": "rain", 
        "confidence": 0.87, 
        "timestamp": "2025-09-29T12:00:00Z",
        "forecast_time": "2025-09-29T18:00:00Z",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-43.7300, -22.8200],  # Zona Oeste do Rio
                [-43.1000, -22.8200],
                [-43.1000, -23.0800],
                [-43.7300, -23.0800],
                [-43.7300, -22.8200]
            ]]
        },
        "location_name": "Zona Oeste do Rio de Janeiro",
        "precipitation_probability": 87,
        "expected_intensity": "moderate"
    },
    {
        "id": 2, 
        "model": "temperature_forecast", 
        "result": 26.1, 
        "confidence": 0.92, 
        "timestamp": "2025-09-29T12:00:00Z",
        "forecast_time": "2025-09-30T12:00:00Z",
        "geometry": {
            "type": "Point",
            "coordinates": [-43.1729, -22.9068]  # UFRJ Campus
        },
        "location_name": "Campus UFRJ - Cidade Universitária",
        "unit": "C",
        "min_temp": 22.5,
        "max_temp": 29.8
    },
    {
        "id": 3,
        "model": "wind_pattern_analysis",
        "result": {
            "direction": "SE",
            "speed_avg": 4.2,
            "speed_max": 7.8,
            "pattern": "stable"
        },
        "confidence": 0.78,
        "timestamp": "2025-09-29T12:00:00Z",
        "forecast_time": "2025-09-29T15:00:00Z",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-43.3000, -22.7000],  # Baía de Guanabara completa
                [-43.0500, -22.7000],
                [-43.0500, -22.9500],
                [-43.3000, -22.9500],
                [-43.3000, -22.7000]
            ]]
        },
        "location_name": "Baía de Guanabara",
        "unit": "m/s"
    },
    {
        "id": 4,
        "model": "severe_weather_alert",
        "result": "thunderstorm_warning",
        "confidence": 0.65,
        "timestamp": "2025-09-29T12:00:00Z",
        "forecast_time": "2025-09-29T20:00:00Z",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-44.0000, -22.5000],  # Região Serrana do RJ
                [-42.8000, -22.5000],
                [-42.8000, -22.0000],
                [-44.0000, -22.0000],
                [-44.0000, -22.5000]
            ]]
        },
        "location_name": "Região Serrana do Estado do Rio de Janeiro",
        "alert_level": "moderate",
        "phenomena": ["thunderstorm", "heavy_rain", "strong_winds"]
    }
]
