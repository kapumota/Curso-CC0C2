# travel_provider.py

import json
import os
from faker import Faker
from faker.providers import BaseProvider
import random

# Cargar ubicaciones soportadas desde el archivo JSON
def load_supported_locations():
    # Obtener el directorio donde se encuentra este archivo
    base_path = os.path.dirname(__file__)
    # Construir la ruta completa al archivo supported_locations.json
    json_file = os.path.join(base_path, 'supported_locations.json')
    
    # Abrir y leer el JSON con las ubicaciones soportadas
    with open(json_file, 'r') as f:
        return json.load(f)

# Guardar en memoria las ubicaciones soportadas (aeropuertos y ciudades)
supported_locations = load_supported_locations()


class TravelProvider(BaseProvider):
    def __init__(self, faker_instance):
        # Usar la instancia de Faker proporcionada
        self.fake = faker_instance

    def flight_lookup(self, departure_city, destination_city, num_options=3):
        """
        Genera una lista de opciones de vuelos entre la ciudad (aeropuerto) de salida
        y la ciudad (aeropuerto) de destino.

        - Valida si ambas ciudades están dentro de los aeropuertos soportados.
        - Incluye precios (en este caso, solo Economy; las otras clases están comentadas).
        """
        # Validar que la ciudad de salida esté soportada
        if departure_city not in supported_locations['airports']:
            return {
                "error": (
                    f"Ciudad de salida no soportada: {departure_city}. "
                    f"Los aeropuertos soportados son {supported_locations['airports']}"
                )
            }

        # Validar que la ciudad de destino esté soportada
        if destination_city not in supported_locations['airports']:
            return {
                "error": (
                    f"Ciudad de destino no soportada: {destination_city}. "
                    f"Los aeropuertos soportados son {supported_locations['airports']}"
                )
            }

        # Validar que salida y destino no sean iguales
        if departure_city == destination_city:
            return {"error": "La ciudad de salida y la ciudad de destino no pueden ser la misma."}
        
        flights = []
        for _ in range(num_options):
            # Seleccionar una aerolínea aleatoria
            airline = random.choice(['Delta', 'United', 'Southwest', 'JetBlue', 'American Airlines'])
            # Generar un número de vuelo aleatorio con prefijo según aerolínea
            flight_number = f"{random.choice(['DL', 'UA', 'SW', 'JB', 'AA'])}{random.randint(100, 9999)}"

            # Generar horarios de salida y llegada (fechas aleatorias en un rango)
            departure_time = self.fake.date_time_between(start_date="now", end_date="+30d")
            arrival_time = self.fake.date_time_between(start_date=departure_time, end_date="+30d")

            # Generar precios aleatorios por clase
            economy_price = round(random.uniform(100, 300), 2)
            # economy_plus_price = round(random.uniform(200, 400), 2)  # (comentado)
            # business_price = round(random.uniform(500, 1000), 2)      # (comentado)

            # Agregar la opción de vuelo al listado
            flights.append({
                'airline': airline,
                'departure_airport': departure_city,
                'destination_airport': destination_city,
                'flight_number': flight_number,
                'departure_time': departure_time,
                'arrival_time': arrival_time,
                'price': economy_price
            })

        # Retornar respuesta con código y listado de opciones
        return {"status_code": 200, "flight_options": flights}

    def hotel_lookup(self, city, num_options=3):
        """
        Genera una lista de opciones de hoteles en la ciudad indicada.

        - Valida si la ciudad está dentro del conjunto de ciudades soportadas para hoteles.
        """
        # Validar que la ciudad esté soportada
        if city not in supported_locations['hotel_cities']:
            return {
                "error": (
                    f"Ciudad no soportada: {city}. "
                    f"Las ciudades soportadas son {supported_locations['hotel_cities']}"
                )
            }

        hotels = []
        for _ in range(num_options):
            # Elegir un nombre de hotel aleatorio
            hotel_name = random.choice(['Hilton', 'Marriott', 'Hyatt', 'Holiday Inn', 'Sheraton'])

            # Generar fechas de check-in y check-out
            check_in = self.fake.date_time_between(start_date="now", end_date="+30d")
            check_out = self.fake.date_time_between(start_date=check_in, end_date="+35d")

            # Calcular precios
            price_per_night = random.uniform(100, 500)
            total_price = price_per_night * random.randint(1, 7)  # Entre 1 y 7 noches

            # Agregar la opción de hotel al listado
            hotels.append({
                'hotel_name': hotel_name,
                'city': city,
                'check_in': check_in,
                'check_out': check_out,
                'price_per_night': round(price_per_night, 2),
                'total_price': round(total_price, 2)
            })

        return hotels

# Inicializar Faker de forma global
fake = Faker()

# Inicializar TravelProvider con la instancia de Faker
travel_provider = TravelProvider(fake)
