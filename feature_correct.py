import pandas as pd
import os
import datetime
import csv
from deepdiff import DeepDiff
from multiprocessing import Pool
from lossrun import JugadRunner
from lossrun import JugadFeatures
import multiprocessing


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def multiprocessor_feature_gen(file):

    newdir = "trainingdata_{}".format(stamp)
    process_dir = os.path.join(HERE, newdir)
    pickle_file = HERE.replace("training/lossrun", "scripts/lossrun/jugad_old/Jugad.pkl")
    raw = file + ".raw.csv"
    print "Generating Raw File for: {}".format(file)
    JugadFeatures.run_java_subprocess_onpdf(features_gen_path, file, process_dir , raw)
    print "Generated"
    print "Generating Feaature File for: {}".format(file)

    pdf_file_name=file
    output_file="/"+file+".txt"
    JugadFeatures.generate_feature_file("/"+pdf_file_name, "/"+raw, output_file, process_dir)
    # input_file_features=pd.read_csv(os.path.join(process_dir,output_file), sep="\t", quoting=csv.QUOTE_NONE)
    new_output_file = file + ".txt"
    print "Generating Preds File for: {}".format(file)
    JugadRunner.run_model(output_file, "/"+new_output_file, process_dir, pickle_file)


def features_generation(path, process_continuation=False):
    directory=(os.listdir(path))

    newdir="trainingdata_{}".format(stamp)
    try:
        os.makedirs(newdir)
    except:
        print "Folder already created"
        pass

    if process_continuation:
        removal=[]
        existing=os.path.join(HERE, newdir)
        existing=(os.listdir(existing))
        for items in directory:
            for new_items in existing:
                if items in new_items[:-4]:
                   removal.append(items)
        for items in removal:
            #print(items)
            try:
                directory.remove(items)
            except:
                pass
    try:
        directory.remove("summary")
        directory.remove('.DS_Store')
        print "Removed Summary and DS_Store"
    except:
        pass

    print directory

    for items in directory:
        if "pdf" not in items:
            directory.remove(items)


    start_time = datetime.datetime.now()
    new_pool = MyPool()
    new_pool.map(multiprocessor_feature_gen, directory)
    end_time = datetime.datetime.now()

    print "\n\nTotal run time :{}".format(end_time - start_time)

    #Removing all raw files
    newdir_list=os.listdir(os.path.join(HERE, newdir))
    for files in newdir_list:
        if ".csv" in files:
            os.remove(os.path.join(HERE, newdir, files))


def couple_files(old_tr, new_tr):

    old = (os.listdir(old_tr))
    new = (os.listdir(new_tr))

    if '.DS_Store' in old:
       old.remove('.DS_Store')
    if '.DS_Store' in new:
        new.remove('.DS_Store')

    matched_entries = []
    for n in new:
        for o in old:
            if n[:-4].lower() in o.lower() and o.count('TRAINING_DATA') > 0:
                matched_entries.append([o, n])
    return matched_entries


def deepdiff(dir_old, dir_new):
    old_check=(os.listdir(dir_old))
    new_check=(os.listdir(dir_new))
    print(len(old_check))
    print(len(new_check))
    couples=couple_files(old_tr, new_tr)
    print(len(couples))
    ddiff_result = {"file_name": [], "result": [], "report": [], "sentence_similarity":[]}
    for items in couples:
        print items
        try:
            t1 = pd.read_csv(old_tr+ "/" + items[0],
                             sep="\t", quoting=csv.QUOTE_NONE)
            t2 = pd.read_csv(new_tr + "/" + items[1],
                             sep="\t", quoting=csv.QUOTE_NONE)
            if len(t1)!= len(t2):
                ddiff_result["file_name"].append(items[1][:-4])
                ddiff_result["result"].append("Different dimensions")
                ddiff_result["report"].append("Length of old file {} new file {}".format(len(t1),len(t2)))
                ddiff_result["sentence_similarity"].append("not_app")
            else:
                dd=DeepDiff(dict(t1.drop("_preds_", axis=1)), dict(t2.drop("_preds_", axis=1)), significant_digits=0)
                ddd=dict(dd)
                if "values_changed" in ddd:
                    ddiff_result["file_name"].append(items[1][:-4])
                    ddiff_result["result"].append("Error Found")
                    #final_report = str(ddd["values_changed"])
                    ddiff_result["report"].append("Values Changed")
                    if "True" in str(t1["sentence"]==t2["sentence"]):
                        ddiff_result["sentence_similarity"].append("Yes")
                    else:
                        ddiff_result["sentence_similarity"].append("No")

                else:
                    ddiff_result["file_name"].append(items[1][:-4])
                    ddiff_result["result"].append("Pass")
                    ddiff_result["report"].append("Pass")
                    ddiff_result["sentence_similarity"].append("Pass")
        except:
            pass
    df = pd.DataFrame(ddiff_result)
    df = df.sort_values(by=["file_name"])
    report_name="features_check_report_{}.csv".format(stamp)
    df.to_csv(HERE+"/"+report_name, index=None, header=True, encoding='utf-8')
    print ("The total number of changes are: {}".format(len(df[df["result"]!="Pass"])))
    return df


def feature_correction(dir_old,dir_new):
    from datetime import datetime
    from re import sub
    from decimal import Decimal

    newdir = "training_data_corrected_{}".format(stamp)
    try:
        os.makedirs(newdir)
    except:
        print("Folder with name {} already created".format(newdir))
        pass

    report_name = "features_check_report_{}.csv".format(stamp)
    report=pd.read_csv(os.path.join(HERE,report_name))
    files=report["file_name"].tolist()
    sentence_similarity=report["sentence_similarity"].tolist()
    for i in range(len(files)):
        for cf in couples:
            if files[i] in cf[1]:
                print "Filename: {}".format(files[i])
                old_df = pd.read_csv(os.path.join(dir_old, cf[0]),
                                 sep="\t", quoting=csv.QUOTE_NONE)

                new_df = pd.read_csv(os.path.join(dir_new,cf[1]),
                                 sep="\t", quoting=csv.QUOTE_NONE)

                print "sentence_similartiy: {}".format(sentence_similarity[i])
                if sentence_similarity[i]== "Yes":
                    new_df["_preds_"]=old_df["_preds_"]
                    if "Unnamed: 0" in new_df.columns:
                        new_df= new_df.drop("Unnamed: 0", axis=1)
                    new_df.to_csv(HERE + "/" + newdir + "/" + files[i] + "_TRAINING_DATA_{}".format(stamp) + ".txt",  sep='\t', index=None, encoding="utf-8" )

                elif sentence_similarity[i]== "Pass":
                    new_df["_preds_"]=old_df["_preds_"]
                    if "Unnamed: 0" in new_df.columns:
                        new_df= new_df.drop("Unnamed: 0", axis=1)
                    new_df.to_csv(HERE + "/" + newdir + "/" + files[i] + "_TRAINING_DATA_{}".format(stamp) + ".txt",  sep='\t', index=None, encoding="utf-8" )

                elif sentence_similarity[i]=="not_app":
                    old_df["status"] = [""]*len(old_df)
                    new_df["_preds_"] = ["no_value"]*len(new_df)
                    for j in range(len(old_df)):
                        for k in range(len(new_df)):
                            try:
                                if ((old_df["sentence"][j]).strip() == (new_df["sentence"][k]).strip()) and (old_df["status"][k] != "changed"):
                                    new_df["_preds_"][k] = old_df["_preds_"][j]
                                    old_df["status"][k] = "changed"
                                try:
                                    if int(old_df["sentence"][j]) == int(new_df["sentence"][k]) and (old_df["status"][k] != "changed"):
                                        new_df["_preds_"][k] = old_df["_preds_"][j]
                                        old_df["status"][k] = "changed"
                                except:
                                    pass
                                try:
                                    new_val=datetime.strptime(new_df["sentence"][k], '%m/%d/%Y')
                                    old_val=datetime.strptime(new_df["sentence"][j], '%m/%d/%Y')
                                    if new_val==old_val and (old_df["status"][k] != "changed"):
                                        new_df["_preds_"][k] = old_df["_preds_"][j]
                                        old_df["status"][k] = "changed"
                                except:
                                    pass
                                try:
                                    new_curr=Decimal(sub(r'[^\d.]', '', new_df["sentence"][k]))
                                    old_curr=Decimal(sub(r'[^\d.]', '', old_df["sentence"][j]))
                                    if new_curr == old_curr and (old_df["status"][k] != "changed"):
                                        new_df["_preds_"][k] = old_df["_preds_"][j]
                                        old_df["status"][k] = "changed"
                                except:
                                    pass

                            except:
                                pass
                    if "Unnamed: 0" in new_df.columns:
                        new_df= new_df.drop("Unnamed: 0", axis=1)
                    new_df.to_csv(HERE+"/"+newdir+"/"+files[i]+ "_TRAINING_DATA_{}".format(stamp) + ".txt", sep='\t', index=None, encoding="utf-8")


def raw_generation():
    from lossrun import JugadFeatures
    input_file_path=os.path.join(HERE, "trainingdata","trainingpdf")
    newdir="raw_output"
    os.makedirs(newdir)
    output_file_path=os.path.join(HERE, newdir)
    input_pdfs=(os.listdir(input_file_path))
    for pdfs in input_pdfs:
        if "pdf" not in pdfs:
            input_pdfs.remove(pdfs)
    for input_file in input_pdfs:
        output_file=input_file+"_raw_"+stamp+".txt"
        JugadFeatures.run_java_subprocess_onpdf(input_file_path, input_file, output_file_path, output_file)
        df = pd.read_csv(output_file_path +"/"+ output_file, sep="\t", quoting=csv.QUOTE_NONE)
        df.to_csv(output_file_path +"/"+ output_file, sep='\t', index=None, encoding="utf-8")


def Unnamed_remover(path):
    items=os.listdir(path)
    for item in items:
        df=pd.read_csv(os.path.join(path, item), sep="\t", quoting=csv.QUOTE_NONE)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df.to_csv(os.path.join(path,item), sep='\t', index=None, encoding="utf-8")


if __name__ == "__main__":
    HERE = os.path.abspath(os.path.dirname(__file__))
    now = datetime.datetime.now()
    stamp="{}{}{}".format(now.year,now.month,now.day)
    # raw_generation()
    features_gen_path= "/Users/Sheel.Saket/d3/lossrun/trainingdata/trainingpdf"
    features_generation(features_gen_path, process_continuation=True)
    # old_tr="/Users/Sheel.Saket/d3/Newer"
    # new_tr="/Users/Sheel.Saket/d3/mvp-hdp/training/lossrun/trainingdata_2018817"
    # couples = couple_files(old_tr, new_tr)
    # deepdiff(old_tr,new_tr)
    # feature_correction(old_tr, new_tr)
    # Unnamed_remover("/Users/Sheel.Saket/Documents/training_data_corrected_2018821")