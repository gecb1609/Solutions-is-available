#Import os module
import os
import xlwt

# Ask the user to enter string to search
def parse():
        search_path = input("Enter the search path : ")
        #input("Enter directory path to search : ")
        file_type = input("Enter File Name : ")
        #input("File Type : ")
        search_str = input("Enter the search string : ")
        #input("Enter the search string : ")
        global cec_feature
        cec_feature = input("CEC Feature to Analyse : ")
        row = 0
        column = 0
        workbook = xlwt.Workbook(encoding = "utf-8")
        sheet1 = workbook.add_sheet("Log Analisys Data")
        global line1
        line1 = []
        #search_str = search_str.encode("utf8")
        
        # Append a directory separator if not already present
        if not (search_path.endswith("/") or search_path.endswith("\\") ): 
                search_path = search_path + "/"
                                                                
        # If path does not exist, set search path to current directory
        if not os.path.exists(search_path):
                search_path ="."
                 
       # Repeat for each file in the directory  
        for fname in os.listdir(path=search_path):

           # Apply file type filter   
           if fname.endswith(file_type):                 
                # Open file for reading
                fo = open(search_path + fname)
                 
        try:              
                # Read the first line from the file UnicodeDecodeError
                line = fo.readline().split("//", 1)
               
                # Initialize counter for line number
                line_no = 1
         
                # Loop until EOF
                while line != '' :
                        # Search for string in line
                     
                       index = line.find(search_str)
                       if ( index != -1) :
                                print("[", line_no, ",", index, "] ", line, sep="")
                                line1.append(line)
                                
                        # Split and put it into excel
                                process_id = line[19:24]
                                print ("Process ID is ", process_id)
                                module_id = line[30:37]
                                print ("Module ID is ", module_id)
                                description = line[40:]
                                print ("Dexcription is ", description)
                                column = column+ 1
                                sheet1.write(row,column,process_id)
                                column = column+ 1
                                sheet1.write(row,column,module_id)
                                column = column+ 1
                                sheet1.write(row,column,description)
                                row = row + 1
                                column = 0
                                
                        # Read next line
                       line = fo.readline()
                        #line1 = []
                        # Increment line counter
                       line_no += 1
                # Close the files
                fo.close()
        except UnicodeDecodeError:
                pass
            

        #print("trace 5")
        workbook.save(search_path + "/log_analisys.xls")

# To check One Touch Play Mandatory Feature from CEC Code Stack

def OneTouchPlay():
        with open("myfile.txt", "w+") as f:
                f.writelines(line1)

        #for msg in line1:
        with open("myfile.txt", "r") as f1:
                found = -1
                msg= f1.read()
                if ("Active Source" in msg and "Image View On" in msg) or ("Active Source" in msg and "Text View On" in msg):
                           #     print ("One Touch Play is PASS")
                        found = True
                else:
                            #    print ("One Touch Play is FAIL")
                         if(found != True):
                                found = False
        if(found == True):
                print("One Touch play is PASS")
        elif(found == False):
                print("One Touch play is FAIL")
        elif(found == -1):
                print("One Touch play is FAIL UNKNOWN")

# To check System Stand by Mandatory Feature from CEC Code Stack

def SyetemStandby():
        with open("myfile.txt", "w+") as f:
                f.writelines(line1)

        #for msg in line1:
        with open("myfile.txt", "r") as f1:
                found = -1
                msg= f1.read()
                if ("Standy" in msg):
                           #     print ("One Touch Play is PASS")
                        found = True
                else:
                            #    print ("One Touch Play is FAIL")
                         if("Standy" in msg and "inactive source" in msg) or ("Standy" in msg and "active source" in msg) or ("Standy" in msg and "Routing Change" in msg) or ("Standy" in msg and "Set Stream Path" in msg):
                                found = False
                         else:
                                 if ( found != 1):
                                         found = False
                         
        if(found == True):
                print("System Standby is PASS")
        elif(found == False):
                print("System Stand by is FAIL")
        elif(found == -1):
                print("System Standby is FAIL UNKNOWN")

def AutoAudioSysWakeup():
           with open("myfile.txt", "w+") as f:
                f.writelines(line1)

        #for msg in line1:
           with open("myfile.txt", "r") as f1:
                found = -1
                msg= f1.read()
                if ("System Audio Mode Request" in msg):
                       found = True

                else:
                        if ( found != 1):
                                found = False
                         
           if(found == True):
                print("Auto Audio Wake up PASS")
           elif(found == False):
                print("Auto Audio Wake up FAIL")
           elif(found == -1):
                print("Auto Audio Wake up FAIL UNKNOWN")

# HDMI Switching Functionality

def HDMI_Switch():
        with open("myfile.txt", "w+") as f:
                f.writelines(line1)

        #for msg in line1:
        with open("myfile.txt", "r") as f1:
                found = -1
                msg= f1.read()
                if ("Routing Change" and "SetStream Path" in msg):
                           #     print ("One Touch Play is PASS")
                        found = True
                else:
                        found = False
                         
        if(found == True):
                print("System Standby is PASS")
        elif(found == False):
                print("System Stand by is FAIL")
        elif(found == -1):
                print("System Standby is FAIL UNKNOWN")

def cec_breakout():
        break_out = input("Check if break Out happenned and enter Yes else No: ")
        with open("myfile.txt", "w+") as f:
                f.writelines(line1)

        #for msg in line1:
        with open("myfile.txt", "r") as f1:
                found = -1
                msg= f1.read()
                if ("Inactive Source" in msg and break_out == "Yes"):
                           #     print ("One Touch Play is PASS")
                        found = True
                else:
                        found = False
                         
        if(found == True):
                print("CEC Break Out  is PASS")
        elif(found == False):
                print("CEC Break Out  is FAIL")
        elif(found == -1):
                print("System Standby is FAIL UNKNOWN")
parse()
#print("trace 5_1")
if (cec_feature == "One Touch Play"):
        OneTouchPlay()

if (cec_feature == "System Standby"):
        SyetemStandby()

if (cec_feature == "HDMI Switch"):
        HDMI_Switch()

if (cec_feature == "AutoAudioSysWakeup"):
        AutoAudioSysWakeup()
if (cec_feature =="cec_breakout"):
        cec_breakout()
#print("trace 6")

