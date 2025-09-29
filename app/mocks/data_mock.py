MOCK_DATA = [
    {
        "id": 1, 
        "type": "temperature", 
        "value": 25.3, 
        "unit": "C", 
        "timestamp": "2025-09-29T12:00:00Z",
        "geometry": {
            "type": "Point",
            "coordinates": [-43.1729, -22.9068]  # Rio de Janeiro - UFRJ
        },
        "location_name": "Campus UFRJ - Cidade Universitária"
    },
    {
        "id": 2, 
        "type": "humidity", 
        "value": 78, 
        "unit": "%", 
        "timestamp": "2025-09-29T12:00:00Z",
        "geometry": {
            "type": "Point",
            "coordinates": [-43.2096, -22.9035]  # Copacabana
        },
        "location_name": "Estação Meteorológica Copacabana"
    },
    {
        "id": 3, 
        "type": "wind_speed", 
        "value": 5.2, 
        "unit": "m/s", 
        "timestamp": "2025-09-29T12:00:00Z",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-43.7964, -22.4818],  # Região da Baía de Guanabara
                [-43.0820, -22.4818],
                [-43.0820, -22.7618],
                [-43.7964, -22.7618],
                [-43.7964, -22.4818]
            ]]
        },
        "location_name": "Região Metropolitana do Rio de Janeiro"
    },
    {
        "id": 4,
        "type": "precipitation",
        "value": 12.5,
        "unit": "mm",
        "timestamp": "2025-09-29T12:00:00Z",
        "geometry": {
            "type": "Point",
            "coordinates": [-43.3751, -22.8959]  # Barra da Tijuca
        },
        "location_name": "Estação Pluviométrica Barra da Tijuca"
    },
    {
        "id": 5,
        "type": "pressure",
        "value": 1013.25,
        "unit": "hPa",
        "timestamp": "2025-09-29T12:00:00Z",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-44.5000, -22.0000],  # Área oceânica próxima ao RJ
                [-42.5000, -22.0000],
                [-42.5000, -24.0000],
                [-44.5000, -24.0000],
                [-44.5000, -22.0000]
            ]]
        },
        "location_name": "Região Oceânica Sudeste"
    }
]
