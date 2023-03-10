from Deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_black_friday_purchase_schema(csv_path):
    """
    black friday purchase schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('black_friday_purchase',
                           attributes=['id', 'gender', 'age', 'occupation',
                                       'city_category', 'stay_in_current_city_years', 'marital_status',
                                       'product_category_1',
                                       'product_category_2', 'product_category_3', 'purchase'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('Black_Friday_Purchase_num'),
                           table_size=163321, primary_key=['id'],
                           ))

    return schema
