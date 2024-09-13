import sqlite3

# label_table
#     - contract_address
#     - ContractName
#     - ABI
#     - Proxy
class ContractLabelDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_all_table()
    
    def create_all_table(self):
        self.create_label_table()

    def create_label_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS label_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                ContractName TEXT NOT NULL,
                ABI TEXT NOT NULL,
                Proxy TEXT NOT NULL        
            )
        '''
        self.__create_table(create_table_query)

    def insert_data_in_label_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'ContractName': json_result['ContractName'],
        #     'ABI':json_result['ABI'],
        #     'Proxy':json_result['Proxy']
        # }
        self.__insert_data('label_table',data_to_insert)

    def query_label_table(self,col_list=[]):
        return self.__query_table('label_table',col_list)

    def __insert_data(self, table_name, data_to_insert):
        columns = ', '.join(data_to_insert.keys())
        placeholders = ', '.join('?' * len(data_to_insert))
        values = tuple(data_to_insert.values())
        self.cursor.execute(f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})', values)
        self.conn.commit()

    def __query_exec(self, query_string):
        self.cursor.execute(query_string)
        rows = self.cursor.fetchall()
        return rows
    
    def __query_table(self,table_name,col_list):
        query_string = ''
        if len(col_list) == 0:
            query_string = f"SELECT * FROM {table_name}"
        else:
            col_string = ','.join(col_list)
            query_string = f"SELECT {col_string} FROM {table_name}"
        rows = self.__query_exec(query_string)
        return rows

    def __create_table(self,create_table_query):
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

# txhash_table
#     - contract_address
#     - tx_hash

# tx_info_table
#     - tx_hash
#     - contract_address
#     - msg_sender
#     - value
#     - gasUsed
#     - nonce
#     - input
#     - blockNumber
#     - transactionIndex

# verify_table
#     - contract_address
#     - verify_tag

# bytecode_table
#     - contract_address
#     - bytecode

# fundsource_table
#     - msg_sender
#     - label
#     - trace_list
#     - hop_count
class DataDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_all_table()
    
    def create_all_table(self):
        self.create_txhash_table()
        self.create_tx_info_table()
        self.create_verify_table()
        self.create_bytecode_table()
        self.create_fundsource_table()

    def create_txhash_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS txhash_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                tx_hash TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)

    def create_tx_info_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS tx_info_table (
                tx_hash TEXT PRIMARY KEY NOT NULL,
                contract_address TEXT NOT NULL,
                msg_sender TEXT NOT NULL,
                value TEXT NOT NULL,
                gasUsed TEXT NOT NULL,
                nonce TEXT NOT NULL,
                input TEXT NOT NULL,
                blockNumber TEXT NOT NULL,
                transactionIndex TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)

    def create_verify_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS verify_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                verify_tag TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)

    def create_bytecode_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS bytecode_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                bytecode TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)

    def create_fundsource_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS fundsource_table (
                msg_sender TEXT PRIMARY KEY NOT NULL,
                label TEXT NOT NULL,
                trace_list TEXT NOT NULL,
                hop_count TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)


    def insert_txhash_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'tx_hash': tx_hash,
        # }
        self.__insert_data('txhash_table',data_to_insert)

    def insert_tx_info_table(self,data_to_insert):
        # data_to_insert = {
        #     'tx_hash': tx_hash,
        #     'contract_address': tx_hash_dic[tx_hash],
        #     'msg_sender': txInfo['from'].lower(),
        #     'value': txInfo['value'],
        #     'gasUsed': txInfo['gas'],
        #     'nonce': txInfo['nonce'],
        #     'input':  txInfo['input'],
        #     'blockNumber': txInfo['blockNumber'],
        #     'transactionIndex': txInfo['transactionIndex'],
        # }
        self.__insert_data('tx_info_table',data_to_insert)

    def insert_verify_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'verify_tag': str(verifyTag),
        # }
        self.__insert_data('verify_table',data_to_insert)

    def insert_bytecode_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'bytecode': bytecode,
        # }
        self.__insert_data('bytecode_table',data_to_insert)

    def insert_fundsource_table(self,data_to_insert):
        # data_to_insert = {
        #     'msg_sender': msg_sender,
        #     'label': lable_result,
        #     'trace_list': ','.join(t_array),
        #     'hop_count' : len(t_array)
        # }
        self.__insert_data('fundsource_table',data_to_insert)


    def query_txhash_table(self,col_list=[]):
        return self.__query_table('txhash_table',col_list)
    
    def query_tx_info_table(self,col_list=[]):
        return self.__query_table('tx_info_table',col_list)

    def query_verify_table(self,col_list=[]):
        return self.__query_table('verify_table',col_list)

    def query_bytecode_table(self,col_list=[]):
        return self.__query_table('bytecode_table',col_list)
    
    def query_fundsource_table(self,col_list=[]):
        return self.__query_table('fundsource_table',col_list)


    def __insert_data(self, table_name, data_to_insert):
        columns = ', '.join(data_to_insert.keys())
        placeholders = ', '.join('?' * len(data_to_insert))
        values = tuple(data_to_insert.values())
        self.cursor.execute(f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})', values)
        self.conn.commit()

    def __query_exec(self, query_string):
        self.cursor.execute(query_string)
        rows = self.cursor.fetchall()
        return rows
    
    def __query_table(self,table_name,col_list):
        query_string = ''
        if len(col_list) == 0:
            query_string = f"SELECT * FROM {table_name}"
        else:
            col_string = ','.join(col_list)
            query_string = f"SELECT {col_string} FROM {table_name}"
        rows = self.__query_exec(query_string)
        return rows

    def __create_table(self,create_table_query):
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

# gigahourse_output_table
#     - contract_address
#     - output 
#     - output_text

# timeout_bytecode_table
#     - contract_address
#     - bytecode

class GigahorseOutputDB:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_all_table()
    
    def create_all_table(self):
        self.create_gigahourse_output_table()
        self.create_timeout_bytecode_table()

    def create_gigahourse_output_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS gigahourse_output_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                output TEXT NOT NULL,
                output_text TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)

    def create_timeout_bytecode_table(self):
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS timeout_bytecode_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                bytecode TEXT NOT NULL
            )
        '''
        self.__create_table(create_table_query)


    def insert_data_gigahourse_output_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'output': output,
        #     'output_text': output_text
        # }
        self.__insert_data('gigahourse_output_table',data_to_insert)

    def insert_timeout_bytecode_table(self,data_to_insert):
        # data_to_insert = {
        #     'contract_address': contract_addr,
        #     'bytecode': bytecode,
        # }
        self.__insert_data('timeout_bytecode_table',data_to_insert)


    def query_gigahourse_output_table(self,col_list=[]):
        return self.__query_table('gigahourse_output_table',col_list)

    def query_timeout_bytecode_table(self,col_list=[]):
        return self.__query_table('timeout_bytecode_table',col_list)


    def __insert_data(self, table_name, data_to_insert):
        columns = ', '.join(data_to_insert.keys())
        placeholders = ', '.join('?' * len(data_to_insert))
        values = tuple(data_to_insert.values())
        self.cursor.execute(f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})', values)
        self.conn.commit()

    def __query_exec(self, query_string):
        self.cursor.execute(query_string)
        rows = self.cursor.fetchall()
        return rows
    
    def __query_table(self,table_name,col_list):
        query_string = ''
        if len(col_list) == 0:
            query_string = f"SELECT * FROM {table_name}"
        else:
            col_string = ','.join(col_list)
            query_string = f"SELECT {col_string} FROM {table_name}"
        rows = self.__query_exec(query_string)
        return rows

    def __create_table(self,create_table_query):
        self.cursor.execute(create_table_query)
        self.conn.commit()

    def close_connection(self):
        self.conn.close()

