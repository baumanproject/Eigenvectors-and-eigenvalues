#%%
import tkinter as tk 
from tkinter import *
import tkinter.scrolledtext as tkscrolled
import sqlite3

def start():
    
    cursor.execute("""CREATE TABLE airlines
                        (number text, destination text, 
                        timeup text, timedown text, places text,averagePrice text)"""
                )
    conn.commit    


#(number, destination, timeup,timedown,places,averagePrice)
def add(event):
        try:        
                arr = [addEntry1.get(),addEntry2.get(),addEntry3.get(),
                addEntry4.get(),addEntry5.get(),addEntry6.get()]
                print("arr = ", arr)
                print(("INSERT INTO airlines VALUES (?,?,?,?,?,?)",arr))
                cursor.execute("INSERT INTO airlines VALUES (?,?,?,?,?,?)", arr)
                conn.commit()
               # print(cursor.fetchall())
                infoLabel.config(text = "Data was added to database")
        except Exception:
                infoLabel.config(text = "Error input for this function")

def spare(result):
        dataStr = ''
        for i in result:
                dataStr+=', '.join(i)
                dataStr+='\n'
        return dataStr

def show(event):
        try:
                if showEntry.get() == 'all':
                        sql = "SELECT * FROM airlines"
                        cursor.execute(sql)
                        results = cursor.fetchall()
                        #conn.commit()
                else:
                        sql = "SELECT * FROM airlines WHERE number = {}".format(showEntry.get())
                        cursor.execute(sql)
                        results = cursor.fetchall()
                        #conn.commit()

                print(results)
                if len(results) == 0:
                        #print(len(cursor.fetchall()))
                        infoLabel.config(text = "Nothing found for this query")
                else:
                        #print(len(cursor.fetchall()))
                        u = spare(results)
                        print(u)
                        TKScrollTXT.delete('1.0', END)
                        TKScrollTXT.insert(tk.INSERT,u)
                        #infoLabel.config(text = u)
        except Exception:
                infoLabel.config(text = "Error input for this function")


def delete(event):
        try:
                str_ = deleteEntry.get()
                avail = "SELECT * FROM airlines WHERE number = {}".format(str_)
                cursor.execute(avail)
                #conn.commit()
                if len(cursor.fetchall()) == 0:
                        raise Exception
                sql = "DELETE FROM airlines WHERE number = {}".format(str_) 
                cursor.execute(sql)
                conn.commit()
                infoLabel.config(text = "{} was deleted from database".format(str_))
        except Exception:
                infoLabel.config(text = "Error input for this function")
#back
def add1(cursor):
        data=[('1','2','3','4','5','6')]
        cursor.executemany("""INSERT INTO airlines VALUES (?,?,?,?,?,?) """,data)
        conn.commit()

conn = sqlite3.connect("mydatabase.db")

#dropTableStatement = "DROP TABLE airlines"
#cursor.execute(dropTableStatement)
try:      
        start()
except Exception:
            print("Data base already exists")

with conn:
        cursor = conn.cursor()


        #dropTableStatement = "DROP TABLE airlines"
        #cursor.execute(dropTableStatement)

        #start()
        root = tk.Tk()
        root.title("Airlines")
        root.geometry('500x500')
        root.resizable(0,0)

#show option
        showButton = Button(root, text = "Show",bg = '#00ffff')
        showButton.bind('<Button-1>',show)
        showButton.place(x = 400, y = 170 )
        showLabel = Label(root,bg = 'white',fg = 'black',text = "SHOW OPTION")
        showLabel.place(x = 385, y = 70)
        showEntry = Entry(root,width = 15)
        showEntry.place(x = 385, y = 120)

#delete option
        deleteButton = Button(root,text = "Delete flight",bg = '#00ffff')
        deleteLabel = Label(root,bg = 'white',fg = 'black',text = "DELETE OPTION")
        deleteEntry = Entry(root,width = 20)
        deleteButton.bind('<Button-1>',delete)
        deleteButton.place(x = 205,y = 170)
        deleteLabel.place(x = 200, y = 70)
        deleteEntry.place(x = 185, y = 120)

#info label
        infoLabel = Label(root,bg = 'magenta',fg = 'black',width = 30)
        infoLabel.place(x = 120,y = 350)
        infoLabel.config(height=5, width=40)

#add option
        addButton = Button(root,text = "Add",bg = '#00ffff')
        addLabel = Label(root,bg = 'white',fg = 'black',text = "ADD OPTION")
        addLabel.place(x = 50, y = 70)



#labels
        addLabel1 = Label(root,bg = 'white',fg = 'black',text = "number ")
        addLabel1.place(x = 5, y = 120)

        addLabe2 = Label(root,bg = 'white',fg = 'black',text = "destination")
        addLabe2.place(x = 5, y = 150)

        addLabel3 = Label(root,bg = 'white',fg = 'black',text = "timeup")
        addLabel3.place(x = 5, y = 180)

        addLabel4 = Label(root,bg = 'white',fg = 'black',text = "timedown")
        addLabel4.place(x = 5, y = 210)

        addLabel5 = Label(root,bg = 'white',fg = 'black',text = "places")
        addLabel5.place(x = 5, y = 240)

        addLabel6 = Label(root,bg = 'white',fg = 'black',text = "averagePrice")
        addLabel6.place(x = 5, y = 270)
#

        addButton.bind('<Button-1>',add)
        addButton.place(x = 55,y = 310)

#entries
        addEntry1 = Entry(root,width = 5)
        addEntry1.place(x = 90, y = 120)

        addEntry2 = Entry(root,width = 5)
        addEntry2.place(x = 90, y = 150)

        addEntry3 = Entry(root,width = 5)
        addEntry3.place(x = 90, y = 180)

        addEntry4 = Entry(root,width = 5)
        addEntry4.place(x = 90, y = 210)

        addEntry5 = Entry(root,width = 5)
        addEntry5.place(x = 90, y = 240)

        addEntry6 = Entry(root,width = 5)
        addEntry6.place(x = 90, y = 270)
#
        #lbox = Listbox(root,selectmode=EXTENDED)
        #lbox.place()
        #lbox.place(x = 170,y = 250)
        #lbox.config(height=5, width=40)
        #scroll = Scrollbar(command=lbox.yview)
        #scroll.pack(side=RIGHT, fill=Y)
        #lbox.config(yscrollcommand=scroll.set)
        TKScrollTXT = tkscrolled.ScrolledText(master = root, width=40, height=5,wrap = tk.WORD)
        # set default text if desired
        #TKScrollTXT.insert(1.0, default_text)
        TKScrollTXT.place(x = 170, y = 250)

        root.mainloop()



#%%


#%%


#%%
