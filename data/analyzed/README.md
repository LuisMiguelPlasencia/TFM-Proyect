# Datasets Procesados - Análisis Estadístico

## Descripción
Datasets del mercado inmobiliario español procesados y listos para análisis de ML.

## Procesamiento Aplicado
- ✅ Eliminación de columnas con >60% valores nulos
- ✅ Remoción de outliers usando criterio IQR
- ✅ Imputación inteligente de valores faltantes
- ✅ Filtro de fechas de construcción válidas (≤2025)
- ✅ Creación de variables derivadas (price_per_m2, property_age, etc.)
- ✅ Adición de columna 'province' para identificación geográfica

## Archivos
- `sales_processed.csv`: Dataset de ventas procesado
- `rental_processed.csv`: Dataset de alquileres procesado

## Columnas Principales
### Sales Dataset
Columnas (18): price, bath_num, room_num, house_type, house_id, m2_real, m2_useful, loc_city, loc_zone, construct_date, garage, province, market_type, price_per_m2, property_age, room_density, luxury_score, space_efficiency

### Rental Dataset
Columnas (29): price, bath_num, room_num, house_type, house_id, m2_real, m2_useful, loc_city, loc_zone, construct_date, air_conditioner, balcony, built_in_wardrobe, chimney, garage, garden, lift, reduced_mobility, storage_room, swimming_pool, terrace, province, market_type, price_per_m2, property_age, room_density, luxury_score, accessibility_score, space_efficiency

## Estadísticas
- **Sales**: 62,018 registros
- **Rental**: 1,576 registros
- **Fecha de procesamiento**: 2025-10-08 09:55:26
