# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import pandas as pd
# import os

# def assign_safety_score(junctions_csv_path, importance_csv_path, output_csv_path):
#     """
#     Assigns a safety score to each junction based on feature importances.
#     """
#     try:
#         junctions_df = pd.read_csv(junctions_csv_path)
#         importance_df = pd.read_csv(importance_csv_path, index_col='Feature')
#         importance_dict = importance_df['Importance'].to_dict()
#     except FileNotFoundError as e:
#         print(f"Error: One or both CSV files not found: {e}")
#         return

#     safety_scores = []
#     for index, row in junctions_df.iterrows():
#         safety_score = 0

#         # Feature mapping and scoring logic
#         feature_values = {
#             'junction_type': row['junction_type'].lower().replace(' ', '_'),
#             'num_roads': str(row['num_roads']),
#             'road_type': row['road_type'].lower().replace(' ', '_'),
#             'oneway': 'one_way' if row['oneway'].lower() == 'yes' else ('two_way' if row['oneway'].lower() == 'no' else 'other' if pd.isna(row['oneway']) else row['oneway'].lower().replace(' ', '_')),
#             'lanes': str(row['lanes']) if pd.notna(row['lanes']) else 'unknown',
#             'maxspeed': str(row['maxspeed']) if pd.notna(row['maxspeed']) else 'unknown', # Keep as string initially
#             'surface': row['surface'].lower().replace(' ', '_') if pd.notna(row['surface']) else 'unknown',
#         }

#         # Add 'types_of_junction_' prefix to junction_type
#         if feature_values['junction_type']:
#             junction_type_feature = f"types_of_junction_{feature_values['junction_type']}"
#             if junction_type_feature in importance_dict:
#                 safety_score += importance_dict[junction_type_feature]

#         # Score based on number of roads (can be expanded with more specific logic)
#         if feature_values['num_roads'] in ['3', '4']:  # Example: higher risk at more complex junctions
#             if 'num_roads_3_or_more' in importance_dict:
#                 safety_score += importance_dict['num_roads_3_or_more'] # You might need to create this feature in your importance CSV

#         # Add 'road_type_' prefix to road_type
#         if feature_values['road_type']:
#             road_type_feature = f"road_type_{feature_values['road_type']}"
#             if road_type_feature in importance_dict:
#                 safety_score += importance_dict[road_type_feature] # You might need to create this feature

#         # Score based on oneway
#         if feature_values['oneway'] in ['one_way']:
#             if 'lanes_or_medians_one_way' in importance_dict:
#                 safety_score += importance_dict['lanes_or_medians_one_way']
#         elif feature_values['oneway'] in ['two_way']:
#             if 'lanes_or_medians_two_way' in importance_dict:
#                 safety_score += importance_dict['lanes_or_medians_two_way']
#         elif feature_values['oneway'] in ['other', 'unknown']:
#             if 'lanes_or_medians_other' in importance_dict:
#                 safety_score += importance_dict['lanes_or_medians_other']
#             elif 'lanes_or_medians_unknown' in importance_dict:
#                 safety_score += importance_dict['lanes_or_medians_unknown']

#         # Score based on number of lanes
#         if feature_values['lanes'] in ['1', '2']:
#             if 'lanes_1_or_2' in importance_dict:
#                 safety_score += importance_dict['lanes_1_or_2'] # You might need to create this feature
#         elif feature_values['lanes'] in ['3', '4', '5']:
#             if 'lanes_3_to_5' in importance_dict:
#                 safety_score += importance_dict['lanes_3_to_5'] # You might need to create this feature
#         elif feature_values['lanes'] == 'unknown':
#             if 'lanes_or_medians_unknown' in importance_dict:
#                 safety_score += importance_dict['lanes_or_medians_unknown']

#         # Score based on maxspeed (you might want to categorize these)
#         max_speed_str = feature_values['maxspeed']
#         if max_speed_str != 'unknown':
#             try:
#                 max_speed = int(max_speed_str)
#                 if max_speed > 60:
#                     if 'maxspeed_high' in importance_dict:
#                         safety_score += importance_dict['maxspeed_high'] # You might need to create this feature
#                 elif max_speed <= 30:
#                     if 'maxspeed_low' in importance_dict:
#                         safety_score += importance_dict['maxspeed_low'] # You might need to create this feature
#             except ValueError:
#                 if 'maxspeed_unknown' in importance_dict:
#                     safety_score += importance_dict['maxspeed_unknown']
#         elif 'maxspeed_unknown' in importance_dict:
#             safety_score += importance_dict['maxspeed_unknown']

#         # Score based on surface type
#         if feature_values['surface'] in ['asphalt']:
#             if 'surface_asphalt' in importance_dict:
#                 safety_score += importance_dict['surface_asphalt'] # You might need to create this feature
#         elif feature_values['surface'] in ['gravel', 'unpaved']:
#             if 'surface_unpaved' in importance_dict:
#                 safety_score += importance_dict['surface_unpaved'] # You might need to create this feature
#         elif feature_values['surface'] == 'unknown':
#             if 'surface_unknown' in importance_dict:
#                 safety_score += importance_dict['surface_unknown'] # You might need to create this feature

#         safety_scores.append(safety_score)

#     junctions_df['safety_score'] = safety_scores
#     junctions_df.to_csv(output_csv_path, index=False)
#     print(f"Safety scores assigned and saved to: {output_csv_path}")

# with DAG(
#     dag_id='assign_junction_safety_score',
#     schedule_interval=None,
#     start_date=datetime(2023, 1, 1),
#     catchup=False,
#     tags=['traffic', 'safety'],
# ) as dag:
#     assign_score_task = PythonOperator(
#         task_id='assign_safety_scores',
#         python_callable=assign_safety_score,
#         op_kwargs={
#             'junctions_csv_path': "/opt/airflow/dags/data/junctions_from_geojson.csv",
#             'importance_csv_path': "/opt/airflow/dags/data/top_feature_importances.csv",
#             'output_csv_path': "/opt/airflow/dags/data/assigned_scores.csv",
#         },
#     )