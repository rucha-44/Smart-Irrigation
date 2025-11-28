import requests
from datetime import datetime

class WeatherService:
    def __init__(self):
        self.api_key = "191e6a8053c17b2b2dc2285019560669" 
        self.base_url_current = "http://api.openweathermap.org/data/2.5/weather"
        self.base_url_forecast = "http://api.openweathermap.org/data/2.5/forecast"

    def _process_weather_item(self, item):
        temp_c = item["main"]["temp"]
        wind_kmh = item["wind"]["speed"] * 3.6
        rainfall = 0.0
        if "rain" in item:
            rainfall = item["rain"].get("1h", item["rain"].get("3h", 0.0))

        condition_api = item["weather"][0]["main"]
        condition_desc = item["weather"][0]["description"].title()
        condition_mapped = "Sunny"
        if "Rain" in condition_api or "Drizzle" in condition_api: condition_mapped = "Rainy"
        elif "Cloud" in condition_api: condition_mapped = "Cloudy"
        elif wind_kmh > 25: condition_mapped = "Windy"

        return {
            "temperature": round(temp_c, 2),
            "humidity": round(item["main"]["humidity"], 2),
            "rainfall": round(rainfall, 2),
            "wind_speed": round(wind_kmh, 2),
            "weather_condition": condition_mapped,
            "condition_desc": condition_desc
        }

    def get_weather_data(self, city):
        try:
            params = {'q': city, 'appid': self.api_key, 'units': 'metric'}
            response = requests.get(self.base_url_current, params=params)
            return self._process_weather_item(response.json())
        except Exception as e:
            return {"error": str(e)}

    def get_forecast_data(self, city):
        try:
            params = {'q': city, 'appid': self.api_key, 'units': 'metric'}
            data = requests.get(self.base_url_forecast, params=params).json()
            daily = []
            seen = set()
            for item in data['list']:
                date = item['dt_txt'].split(' ')[0]
                if date not in seen and "12:00" in item['dt_txt']:
                    proc = self._process_weather_item(item)
                    dt = datetime.strptime(date, '%Y-%m-%d')
                    proc['day_name'] = dt.strftime('%A'); proc['date_short'] = dt.strftime('%d %b')
                    daily.append(proc); seen.add(date)
                    if len(daily) >= 4: break
            return daily
        except Exception as e:
            return {"error": str(e)}