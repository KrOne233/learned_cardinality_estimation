from Deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_tpch_schema(csv_path):
    schema = SchemaGraph()

    # tables

    # lineitem
    schema.add_table(Table('lineitem', attributes=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
                                                   'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag',
                                                   'l_linestatus',
                                                   'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct',
                                                   'l_shipmode', 'l_comment', 'l_ps_id', 'l_key'],
                           primary_key=['l_key'],
                           keep_fk_attributes=['l_orderkey', 'l_ps_id'],
                           irrelevant_attributes=['l_returnflag', 'l_linestatus',
                                                  'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct',
                                                  'l_shipmode', 'l_comment'],
                           no_compression=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber',
                                           'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax'],
                           csv_file_location=csv_path.format('lineitem'),
                           table_size=6001215))

    # orders
    schema.add_table(Table('orders', attributes=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate',
                                                 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'],
                           csv_file_location=csv_path.format('orders'),
                           primary_key=['o_orderkey'],
                           keep_fk_attributes=['o_custkey'],
                           irrelevant_attributes=['o_orderstatus', 'o_orderdate', 'o_orderpriority', 'o_clerk',
                                                  'o_shippriority', 'o_comment'],
                           no_compression=['o_custkey', 'o_totalprice'],
                           table_size=1500000))

    # customer
    schema.add_table(Table('customer', attributes=['c_custkey', 'c_name', 'c_address', 'c_nationkey',
                                                     'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'],
                           csv_file_location=csv_path.format('customer'),
                           primary_key=['c_custkey'],
                           keep_fk_attributes=['c_nationkey'],
                           irrelevant_attributes=['c_name', 'c_address', 'c_phone', 'c_mktsegment', 'c_comment'],
                           no_compression=['c_nationkey', 'c_acctbal'],
                           table_size=150000))

    # partsupp
    schema.add_table(Table('partsupp', attributes=['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost',
                                                   'ps_comment', 'ps_id'],
                           csv_file_location=csv_path.format('partsupp'),
                           primary_key=['ps_id'],
                           keep_fk_attributes=['ps_partkey', 'ps_suppkey'],
                           irrelevant_attributes=['ps_comment'],
                           no_compression=['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost'],
                           table_size=800000))

    # part
    schema.add_table(Table('part', attributes=['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size',
                                               'p_container', 'p_retailprice', 'p_comment'],
                           csv_file_location=csv_path.format('part'),
                           primary_key=['p_partkey'],
                           irrelevant_attributes=['p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_container', 'p_comment'],
                           no_compression=['p_size', 'p_retailprice'],
                           table_size=200000))

    # region
    schema.add_table(Table('region', attributes=['r_regionkey', 'r_name', 'r_comment'],
                           csv_file_location=csv_path.format('region'),
                           primary_key=['r_regionkey'],
                           irrelevant_attributes=['r_name', 'r_comment'],
                           # no_compression=[],
                           table_size=5))

    # supplier
    schema.add_table(Table('supplier', attributes=['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone',
                                                   's_acctbal', 's_comment'],
                           primary_key=['s_suppkey'],
                           csv_file_location=csv_path.format('supplier'),
                           keep_fk_attributes=['s_nationkey'],
                           irrelevant_attributes=['s_name', 's_address', 's_phone', 's_comment'],
                           no_compression=['s_nationkey', 's_acctbal'],
                           table_size=10000))

    # nation
    schema.add_table(Table('nation', attributes=['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'],
                           csv_file_location=csv_path.format('nation'),
                           primary_key=['n_nationkey'],
                           keep_fk_attributes=['n_regionkey'],
                           irrelevant_attributes=['n_name', 'n_comment'],
                           no_compression=['n_regionkey'],
                           table_size=25))

    # relationships
    # Currently, only single primary keys are supported for table with incoming edges
    schema.add_relationship('lineitem', 'l_orderkey', 'orders', 'o_orderkey')
    schema.add_relationship('lineitem', 'l_ps_id', 'partsupp', 'ps_id')
    schema.add_relationship('orders', 'o_custkey', 'customer', 'c_custkey')
    schema.add_relationship('partsupp', 'ps_partkey', 'part', 'p_partkey')
    schema.add_relationship('partsupp', 'ps_suppkey', 'supplier', 's_suppkey')
    schema.add_relationship('supplier', 's_nationkey', 'nation', 'n_nationkey')
    schema.add_relationship('customer', 'c_nationkey', 'nation', 'n_nationkey')
    schema.add_relationship('nation', 'n_regionkey', 'region', 'r_regionkey')

    return schema
