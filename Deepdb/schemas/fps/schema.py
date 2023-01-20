from Deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_fps_schema(csv_path):
    """
    fps schema
    """

    schema = SchemaGraph()
    schema.add_table(Table('fps',
                           attributes=['id', 'cpuname', 'cpunumberofcores', 'cpunumberofthreads',
                                       'cpubaseclock', 'cpucachel1', 'cpucachel2', 'cpucachel3', 'cpudiesize',
                                       'cpufrequency', 'cpumultiplier', 'cpumultiplierunlocked',
                                       'cpuprocesssize', 'cputdp', 'cpunumberoftransistors', 'cputurboclock',
                                       'gpuname', 'gpuarchitecture', 'gpubandwidth', 'gpubaseclock',
                                       'gpuboostclock', 'gpubusnterface', 'gpunumberofcomputeunits',
                                       'gpudiesize', 'gpudirectx', 'gpunumberofexecutionunits',
                                       'gpufp32performance', 'gpumemorybus', 'gpumemorysize', 'gpumemorytype',
                                       'gpuopencl', 'gpuopengl', 'gpupixelrate', 'gpuprocesssize',
                                       'gpunumberofrops', 'gpushadermodel', 'gpunumberofshadingunits',
                                       'gpunumberoftmus', 'gputexturerate', 'gpunumberoftransistors',
                                       'gpuvulkan', 'gamename', 'gameresolution', 'gamesetting', 'dataset',
                                       'fps'],
                           irrelevant_attributes=['gpuarchitecture', 'gpuopencl', 'gpuopengl', 'gpudirectx', 'dataset'],
                           csv_file_location=csv_path.format('fps_num_lower'),
                           table_size=425833, primary_key=['id'],
                           ))

    return schema
