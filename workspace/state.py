from dataclasses import dataclass, field 
import copy

# The dataclass import allows a simpler process of building up a new data type, instead of rewriting 
# things like the constructor, dataclass does it automatically for you. As for field, it is the function
# with default_factory gives each instance its own fresh mutable default, instead of all instances sharing one. 
# If you make the default value for list[str] to be [], every email you make would share that same single list, 
# so field allows for a fresh list per object.

# As for copy, that import holds the deepcopy function we will need because every time we reset the environment
# we need a fresh slate that holds the untouched copy of the World before the agent runs through it. Deepcopy as 
# opposed to doing copy = world, is different because those two point at the same object still whereas deepcopy
# makes an actual pristine copy of the world allowing you to also compare it to the world at the end of an agent run
# for testing and verifying purposes.

@dataclass
class Attachment:
    attachment_id: str
    name: str
    content: str

# This is the Attachment class and it is used to represent the attachment datatype that the agent can access. It includes
# attachement_id (string), name (string), content (string). The reason we are making this its own class instead of including
# it into the file type is because it does not have a folder and it lives within the email making it simpler to include as its own class.

@dataclass
class Email:
    email_id: str
    sender: str
    subject: str
    body: str
    thread_id: str | None = None
    recipients: list[str] = field(default_factory = list)
    folder: str = "Inbox"
    read: bool = False
    attachments: list[Attachment] = field(default_factory = list)

# This is the Email class and it is used to represent the email datatype that the agent can access. It holds email_id (string),
# thread_id (string and None if not defined), sender (string), subject (string), body (string), recipients (list of strings), 
# read (boolean), attachments (list of Attachment datatype) and folder (string).

@dataclass
class DriveFile:
    file_id: str
    name: str
    content: str
    folder_path: str
    shared_with: list[str] = field(default_factory = list)

# The one of the four sub dataclasses is the DriveFile class which will be representing the file drive aspect of the world. It includes
# file_id (string), name (string), content (string), folder_path (string), and a crucial shared_with (list of strings) to introduce
# a new aspect to tasks and ensure the rubric is satisfied. 

@dataclass
class World:
    emails: dict[str, Email] = field(default_factory = dict)
    files: dict[str, DriveFile] = field(default_factory = dict)
    folders: set[str] = field(default_factory=set)
    id_counter: int = 0
    clarifications: list[str] = field(default_factory=list)  
    done_answer: str | None = None

    # These are the cornerstone datatypes that will be seen throughout the world. Emails representing a dictionary with email_id -> Email,
    # files being a dictionary with file_id -> DriveFile and folders being a set of strings representing folder paths where nesting is represented by slashes.
    # THe id_counter is simply an integer which can be used for id generation. Clarification is just used to store questions the agent asks for verifying purposes
    # and done_answer is just a string of whatever answer the agent might add after the task is done.

    def next_id(self, prefix: str) -> str:
        self.id_counter += 1
        return f"{prefix}_{self.id_counter:03d}" 

    # This function can be used to generate new unique ids e.g. email_001 or file_005

    def get_file(self, file_id: str) -> DriveFile | None:
        return self.files.get(file_id)

    # Returns the file based on file_id from the files dictionary

    def folder_exists(self, path: str) -> bool:
        return path in self.folders

    # Checks whether a specific path exists in the folders set

    def search_files(self, query: str) -> list[str]:
        q = query.lower()
        return [fid for fid, f in self.files.items() 
                if q in f.name.lower() 
                or q in f.content.lower()]
    
    # Searches through each file and id pairing for the query by looking at the name or content of the file. 

    def get_email(self, email_id: str) -> Email | None:
        return self.emails.get(email_id) 

    # Returns the email based on email_id from the emails dictionary.

    def search_emails(self, query: str) -> list[str]:
        q = query.lower()
        return [eid for eid, e in self.emails.items()
                if q in e.subject.lower()
                or q in e.body.lower()
                or q in e.sender.lower()] 
    
    # Searches through each email and id pairing for the query by looking at if the query is in the sender, subject or body

    def get_thread(self, thread_id: str) -> list[Email]:
        return [e for e in self.emails.values() if e.thread_id == thread_id]

    # Searches through each email in the dictionary and returns a list of all emails that have a specific thread_id, because
    # emails that share the same thread_id are apart of the same conversation.
    
    def snapshot(self) -> "World":
        return copy.deepcopy(self) 

    # Allows for a snapshot of the world in its current state to be taken.
    
@dataclass
class Task:
    instruction: str
    world: World
    goal_spec: list[dict]
    metadata: dict = field(default_factory=dict)

# This is the task subclass which is to define how a task will look. It includes the instruction string, the world we are operating on itself,
# then we have goal_spec which is essentially the answer key and metadata is a dict of thtings that can affect the tasks like seed or difficulty.


# The general purpose of this file is to define the World, a place where the objecst of emails and drive files live. The four
# building blocks of the workspace include the attachment datatype class, the email one and the drive file one whereas World
# serves as the single container holding them all, defining what exists and how you can access the information in it. 