def transformToDictList(data: list):
    return list(map(lambda x: dict(x), data))

def queryResultToDict(query_result,related_tables=None):
    result_list = []
    for row in query_result:
        row_dict = {}
        for column in row.__table__.columns:
            row_dict[column.name] = getattr(row, column.name)

        # Handle related tables
        if related_tables:
            for related_table in related_tables:
                related_data = getattr(row, related_table, None)
                if related_data is not None:
                    if isinstance(related_data, list):
                        related_data = queryResultToDict(related_data)
                    else:
                        related_data = queryResultToDict([related_data])[0]
                    row_dict[related_table] = related_data
        
        result_list.append(row_dict)
    return result_list