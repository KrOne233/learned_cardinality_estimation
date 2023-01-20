from Deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_black_friday_purchase_schema(csv_path):
    """
    black friday purchase schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('black_friday_purchase',
                           attributes=['id', 'Gender', 'Age', 'Occupation',
                                       'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status',
                                       'Product_Category_1',
                                       'Product_Category_2', 'Product_Category_3', 'Purchase'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('Black_Friday_Purchase_num'),
                           table_size=163321, primary_key=['id'],
                           ))

    return schema
