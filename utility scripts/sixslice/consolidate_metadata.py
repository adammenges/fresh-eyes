import os, time

DELETE_FILES = True

def main():
    basepath = "W:\\Box Sync\\SG 2018 Datasets\\180314 fourview - 100"
    do_process_root = True
    tic = time.clock()

    collected_metadata = []
    for root, dirs, files in walklevel(basepath):
        if (not do_process_root) and (root == basepath): continue
        for full_filename in files:
            filename, file_extension = os.path.splitext(full_filename)
            if file_extension != ".txt": continue
            if filename+".png" not in files:
                print("Orphan file found.\t{}\t{}".format(os.path.basename(root),filename))
                continue
                
            with open(os.path.join(root,full_filename)) as f: 
                content = [x.strip() for x in f.readlines()][0] # ONLY READS FIRST LINE OF TEXT FILE
            collected_metadata.append(content+"\n")
            if (DELETE_FILES): os.remove(os.path.join(root,full_filename))
    
    collected_metadata[-1] = collected_metadata[-1].strip()
    f = open(os.path.join(basepath,"_metadata.csv"), "w")
    f.writelines(collected_metadata)
    f.close()
    
    t = round(time.clock()-tic)
    print("time:\t{}s".format(t))



def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), "Directory not found:\t{}".format(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

if __name__ == "__main__":
    # execute only if run as a script
    main()