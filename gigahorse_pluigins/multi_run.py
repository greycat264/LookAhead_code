import sqlite3
import time
import os
import shutil
import subprocess
from db_tool import DataDB,GigahorseOutputDB
import concurrent.futures


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            # print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error: {e}")
    else:
        # print(f"Folder '{folder_path}' already exists.")
        pass

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e}")

def list_directories(path):

    contents = os.listdir(path)

    directories = [content for content in contents if os.path.isdir(os.path.join(path, content))]
    return directories

def getInputBytecodeDic(db_name,del_items=[]):
    result_dic = {}
    dataDB = DataDB(db_name)
    rows = dataDB.query_bytecode_table()
    for row in rows:
        if row[0] not in del_items:
            result_dic[row[0]] = row[1]
    dataDB.close_connection()
    return result_dic


def getBlocknumberDic(db_name):
    result_dic = {}
    dataDB = DataDB(db_name)
    rows = dataDB.query_tx_info_table(['contract_address','blockNumber'])
    for row in rows:
        result_dic[row[0]] = int(row[1],16)
    dataDB.close_connection()
    return result_dic

def exec_codetext(cmd):
    # print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    return process.returncode

def run_my_gigahorse(dataset_dir,output_folder_path,max_workers):
    ok_result_dic = {}
    start_time = time.time()

    # process = subprocess.Popen(f"python3 gigahorse.py -C clients/features_client.dl {dataset_dir} -w {output_folder_path} --restart -T 10", shell=True)
    process = subprocess.Popen(f"python3 gigahorse.py -C clients/features_client.dl {dataset_dir} -w {output_folder_path} --restart -T 30 -j 60>> /dev/null 2>&1", shell=True)
    # process = subprocess.Popen(f"python3 gigahorse.py -C clients/features_client.dl {dataset_dir} -w {output_folder_path} --restart -T 30 -j 48", shell=True)
    # process = subprocess.Popen(f"python3 gigahorse.py -C clients/features_client.dl {dataset_dir} -w {output_folder_path} --restart -T 2000 >> /dev/null 2>&1", shell=True)
    process.wait()

    end_time = time.time()
    execution_time = end_time - start_time

    directories = list_directories(output_folder_path)

    cmd_list = []
    for directory in directories:
        file_path1 = f'{output_folder_path}/{directory}/out/AllFeature.csv'
        if os.path.exists(file_path1):
            blocknumber = BLOCKNUMBER_DIC[directory]
            cmd = f"python3 codetext.py {directory} {blocknumber} {output_folder_path}"
            cmd_list.append(cmd)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for cmd  in cmd_list:
            results.append(executor.submit(exec_codetext, cmd))
        
        for future in concurrent.futures.as_completed(results):
            r = future.result() 
            if r == 111:
                pass
            else:
                print(f'return code ==> {r}')

    end_time = time.time()
    execution_time = end_time - start_time


    for directory in directories:
        file_path1 = f'{output_folder_path}/{directory}/out/AllFeature.csv'
        file_path2 = f'{output_folder_path}/{directory}/out/codetext.tac'

        if os.path.exists(file_path1) and os.path.exists(file_path2):
            contract_address = directory.split('.')[0]
            with open(file_path1,'r') as f:
                output = f.readline().strip()
            with open(file_path2,'r') as f:
                output_text = f.read()
            data = {
                'output': output,
                'output_text': output_text
            }
            ok_result_dic[contract_address] = data
        else:
            if not os.path.exists(file_path1):
                print(f'{file_path1} not found')
            if not os.path.exists(file_path2):
                print(f'{file_path2} not found')
        #     exit(0)
    return ok_result_dic

def main():
    # input_db = 'attack/evm/evm_data.db'
    # output_db = 'gigahorse_output_evm.db'
    input_db = 'attack/0day/evm_data.db'
    output_db = 'gigahorse_output_0day.db'

    gigahorse_output_db = GigahorseOutputDB(output_db)
    gigahorse_output_db.create_all_table()

    timeout_contract_list = [item[0] for item in gigahorse_output_db.query_timeout_bytecode_table(['contract_address'])]
    done_contact_list = [item[0] for item in gigahorse_output_db.query_gigahourse_output_table(['contract_address']) ]
    del_list = timeout_contract_list + done_contact_list

    bytecode_dic = getInputBytecodeDic(input_db,del_list)


    global BLOCKNUMBER_DIC
    BLOCKNUMBER_DIC = getBlocknumberDic(input_db)
    print(len(BLOCKNUMBER_DIC.keys()))
    # exit()

    dataset_dir = 'gigahorse_dataset'
    output_folder_path = "myoutput"
    batch = 500

    create_folder_if_not_exists(dataset_dir)

    contract_address_list = list(bytecode_dic.keys())
    round = 0
    while True:
        if len(contract_address_list) <= batch*(round+1) and round != 0:
            break
        
        run_list = contract_address_list[ round*batch : batch+round*batch ]
        print(f'round {round+1} , [{round*batch}~{batch+round*batch}]')


        delete_folder(dataset_dir)
        create_folder_if_not_exists(dataset_dir)
        for contract_addr in run_list:
            bytecode = bytecode_dic[contract_addr][2:]
            with open(f'{dataset_dir}/{contract_addr}.hex','w+') as f:
                f.write(bytecode)

        s_time = time.time()

        gigahorse_result_dic = run_my_gigahorse(dataset_dir,output_folder_path,max_workers=5)
        e_time = time.time()
        execution_time = e_time - s_time


        ok_list = list(gigahorse_result_dic.keys())
        timeout_list = [item for item in run_list if item not in ok_list]

        


        for contract_addr in ok_list:
            data_to_insert = {
                'contract_address': contract_addr,
                'output': gigahorse_result_dic[contract_addr]['output'],
                'output_text' : gigahorse_result_dic[contract_addr]['output_text']
            }
            gigahorse_output_db.insert_data_gigahourse_output_table(data_to_insert)
        for contract_addr in timeout_list:
            data_to_insert = {
                'contract_address': contract_addr,
                'bytecode': bytecode_dic[contract_addr],
            }
            gigahorse_output_db.insert_timeout_bytecode_table(data_to_insert)
        round = round + 1
        # exit(0)

    gigahorse_output_db.close_connection()
try:
    main()
except Exception as e:
    print(e)
    exit(0)
