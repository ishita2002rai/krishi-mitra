"""Local weather lookup via Open-Meteo (free, no API key) for Increment 3's
disease-spread risk reasoning."""

import requests

from krishi_mitra.config import FORECAST_DAYS

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _to_place(result: dict) -> dict:
    return {
        "name": result["name"],
        "admin1": result.get("admin1", ""),
        "country": result.get("country", ""),
        "lat": result["latitude"],
        "lon": result["longitude"],
    }


def search_locations(query: str, count: int = 5) -> list:
    """Up to `count` candidate places matching a partial location name, for
    disambiguating/autocompleting a farmer's location input (see app.py)."""
    if len(query.strip()) < 3:
        return []
    response = requests.get(GEOCODING_URL, params={"name": query, "count": count}, timeout=10)
    response.raise_for_status()
    return [_to_place(r) for r in response.json().get("results") or []]


def geocode(location: str) -> dict:
    """Resolve a free-text location (e.g. 'Pune, India') to lat/lon via Open-Meteo."""
    response = requests.get(GEOCODING_URL, params={"name": location, "count": 1}, timeout=10)
    response.raise_for_status()
    results = response.json().get("results")
    if not results:
        raise SystemExit(f"Could not find a location matching '{location}'. Try a nearby city name.")
    return _to_place(results[0])


def format_place(place: dict) -> str:
    """Display string for a place dict, e.g. 'Pune, Maharashtra, India'."""
    parts = [place["name"], place.get("admin1", ""), place.get("country", "")]
    return ", ".join(p for p in parts if p)


def fetch_weather_summary(location) -> dict:
    """Current conditions + a short-term daily forecast for a location, which may be
    a free-text string (geocoded here) or an already-resolved place dict (from
    geocode()/search_locations()), to avoid re-geocoding a farmer's disambiguated pick."""
    place = location if isinstance(location, dict) else geocode(location)
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": place["lat"],
            "longitude": place["lon"],
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    daily = [
        {
            "date": data["daily"]["time"][i],
            "temp_max_c": data["daily"]["temperature_2m_max"][i],
            "temp_min_c": data["daily"]["temperature_2m_min"][i],
            "precipitation_mm": data["daily"]["precipitation_sum"][i],
            "precipitation_probability_pct": data["daily"]["precipitation_probability_max"][i],
        }
        for i in range(len(data["daily"]["time"]))
    ]

    return {
        "location": format_place(place),
        "current": {
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "precipitation_mm": current["precipitation"],
        },
        "daily": daily,
    }


def format_summary(weather: dict) -> str:
    """Plain-text rendering of a weather summary, for prompts and CLI output."""
    lines = [f"Weather in {weather['location']}:"]
    c = weather["current"]
    lines.append(f"  Now: {c['temperature_c']}°C, {c['humidity_pct']}% humidity, {c['precipitation_mm']}mm precipitation")
    for day in weather["daily"]:
        lines.append(
            f"  {day['date']}: {day['temp_min_c']}-{day['temp_max_c']}°C, "
            f"{day['precipitation_mm']}mm precip ({day['precipitation_probability_pct']}% chance)"
        )
    return "\n".join(lines)
