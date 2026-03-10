import os
import json
import pandas as pd
from  webdav3.client import Client
from pathlib import Path 
from datetime import datetime
import logging

OPTIONS = {
 'webdav_hostname': "",
 'webdav_login':    "",
 'webdav_password': "",
 'webdav_verbose': True,}

def create_structure():
    for i in range(1, 29):
        submissions_folder_path = "./EDIH_AI4Health_Challenge/" + "Tim_{:02d}".format(i)
        if not os.path.exists(submissions_folder_path):
            os.makedirs(submissions_folder_path)
            
            df = pd.DataFrame(columns=["Team", "Date", "Order", 
                    "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"])

            df.to_csv(os.path.join(submissions_folder_path, 'metrics_public.csv'))
            df.to_csv(os.path.join(submissions_folder_path, 'metrics_private.csv'))

def collect_files(client, log, submissions_folder_path="/EDIH_AI4Health_Challenge/", 
            download_folder_path="./EDIH_AI4Health_Challenge"):
    team_folders = ["Tim_{:02d}".format(i) for i in range(1, 29)]  
    team_files = {}    
    for team_folder in team_folders:
        # Path to the Submission folder for the current team
        submissions_folder = submissions_folder_path + f"/{team_folder}/Submission/"
        contents = client.list(submissions_folder)[1:]  # 0 - folder name, skip it
        log.info(f"Contents of {submissions_folder}: {contents}")
        # Create a local folder for the team if it doesn't exist
        team_local_folder = os.path.join(download_folder_path, team_folder)
        if not os.path.exists(team_local_folder):
            os.makedirs(team_local_folder)
        # Download files to the local folder
        for item in contents:
            remote_item_path = os.path.join(submissions_folder, item)
            local_item_path = os.path.join(team_local_folder, item)
            local_csv = local_item_path.replace('.txt', '.csv')
            if os.path.isfile(local_item_path):
                log.info(f"File {item} already exists in {team_local_folder}.")
            elif os.path.isfile(local_csv):
                log.info(f"File {item} already exists in {team_local_folder}.")                
            else:
                log.info(client.info(remote_item_path))
                if local_item_path.endswith('.csv'):
                    if os.path.isfile(local_item_path):
                        pass
                    else:
                        client.download_async(remote_item_path, local_item_path)
                else:
                    item_csv = item.replace('.txt', '.csv')
                    local_item_path = os.path.join(team_local_folder, item_csv)
                    if os.path.isfile(local_item_path):
                        pass
                    else:
                        client.copy(remote_item_path, f'./{item_csv}')
                        client.download_async(f'./{item_csv}', local_item_path)
                log.info(f"DOWNLOADING {team_folder}: {local_item_path},  downloaded.")

        # Store the list of downloaded items for the team
        log.info(f"Team {team_folder} has {len(contents)} files.")
        team_files[team_folder] = contents
        with open('team_files.json', 'w') as f:
            json.dump(team_files, f)

    return team_files

def push_file_to_cloud(client, log, submissions_folder_path="/EDIH_AI4Health_Challenge/", 
                    local_f_p = "EDIH_AI4Health_Challenge",
                    datoteka= 'metrics_public.csv'):
    team_folders = ["Tim_{:02d}".format(i) for i in range(1, 29)]  # List of team folders
    datoteka = datoteka 
    for team_folder in team_folders:
        # print(team_folder)
        try:
            # Define the local path to the folder for the current team
            local_folder_path = os.path.join(local_f_p, team_folder)
            local_file_path = os.path.join(local_folder_path, datoteka)            
            if os.path.isfile(local_file_path):
                # Define the remote path to the Submission folder for the current team
                remote_folder_path = local_folder_path
                remote_csv_path = local_file_path
                remote_csv_path = remote_csv_path.replace('\\', '/')
                # if client.info(remote_csv_path):
                client.clean(remote_csv_path)
                log.info(remote_csv_path)
                client.upload(remote_csv_path, local_file_path)
                log.info(f"Uploaded {datoteka} from {team_folder} to cloud {remote_csv_path}.")
        except Exception as e:
            log.info(f"Error occurred while processing {team_folder}: {e}")

def collect_files_final_eval(client, log, submissions_folder_path="/EDIH_AI4Health_Challenge/", 
            download_folder_path="./EDIH_AI4Health_Challenge"):
    team_folders = ["Tim_{:02d}".format(i) for i in range(1, 29)]  
    team_files = {}    
    for team_folder in team_folders:
        submissions_folder = submissions_folder_path + f"/{team_folder}/Submission/"
        # List contents of the Submission folder
        contents = client.list(submissions_folder)[1:]  # 0 - folder name, skip it
        log.info('*'*20, team_folder, '*'*20)
        log.info(f"Team {team_folder} has {len(contents)} files.")
        team_local_folder = os.path.join(download_folder_path, team_folder)
        team_local_final = os.path.join(download_folder_path, team_folder, 'final')
        os.makedirs(team_local_final, exist_ok=True)
        date_string = 'Mon, 04 Mar 2024 08:04:55 GMT'
        date0 = datetime.strptime(
                    date_string, '%a, %d %b %Y %H:%M:%S %Z')
        remote_final_item = None
        for item in contents:
            remote_item_path = os.path.join(submissions_folder, item)
            local_item_path = os.path.join(team_local_final, item)
            file_dict = client.info(remote_item_path)
            datetime_object = datetime.strptime(
                file_dict['modified'], '%a, %d %b %Y %H:%M:%S %Z')
            
            if datetime_object > date0:
                date0 = datetime_object
                remote_final_item = remote_item_path
                final_local = local_item_path
                final_item = item
        if remote_final_item is None:
            pass
        else:
            log.info('Final file to download: {} \n Client info:'.format(
                remote_final_item), client.info(remote_final_item))
        
            if final_item.endswith('.csv'):
                client.download_async(remote_final_item, final_local)
            else:
                item_csv = final_item.replace('.txt', '.csv')
                team_local_final = os.path.join(team_local_final, item_csv)
                if os.path.isfile(team_local_final):
                    pass
                else:
                    client.copy(remote_final_item, f'./{item_csv}')
                    client.download_async(f'./{item_csv}', team_local_final)
            log.info(f"DOWNLOADING {remote_final_item}:--> {team_local_final}, downloaded.")


if __name__ == "__main__":
    date = datetime.now().strftime("%d%m%y")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"getFilesCloud_{date}.log"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger(__name__)
    client = Client(OPTIONS)
    collect_files(client, log)
    push_file_to_cloud(client, log)
    collect_files_final_eval(client, log)
    