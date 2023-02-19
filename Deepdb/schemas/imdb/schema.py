from Deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                                  'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=2528312))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           keep_fk_attributes=['movie_id'],
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=1380035))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id'],
                           csv_file_location=csv_path.format('movie_info'),
                           keep_fk_attributes=['movie_id'],
                           irrelevant_attributes=[],
                           no_compression=['info_type_id'],
                           table_size=14835720))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           keep_fk_attributes=['movie_id'],
                           irrelevant_attributes=[],
                           no_compression=['role_id', 'person_id'],
                           table_size=36244344))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           keep_fk_attributes=['movie_id'],
                           no_compression=['keyword_id'],
                           table_size=4523930))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           keep_fk_attributes=['movie_id'],
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=2609129))

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema



