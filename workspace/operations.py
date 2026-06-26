from enum import Enum
from workspace.state import World, Email, DriveFile, Attachment
from typing import Any

# We are importing Enum because we use it to define a fixed, closed list of every legal action name rather than raw, 
# susceptible to typo, strings. As for the .state import, we are importing the data classes we defined in the state file
# and the World class we built that defines how the world looks and can be accessed. 

invalid_action_penalty = -0.05
step_cost = -0.02

# The invalid_action_penalty prevents the agent from spamming actions that are not valid like calling tools that don't
# exist or trying to move a nonexistent file. THe step_cost prevents the agent from wandering and forces it to complete
# the task in the most efficient way. 

class ActionType(str, Enum):
    SEARCH_EMAIL = "search_email"
    OPEN_EMAIL = "open_email"

    # These are the reading based actions relating to emails

    SEND_EMAIL = "send_email"
    REPLY_EMAIL = "reply_email"
    FORWARD_EMAIL = "forward_email"
    ARCHIVE_EMAIL = "archive_email"

    # These are the writing based actions relating to emails

    SEARCH_DRIVE = "search_drive"
    OPEN_FILE = "open_file"

    # These are reading based actions relating to files

    MOVE_FILE = "move_file"
    COPY_FILE = "copy_file"
    RENAME_FILE = "rename_file"
    DELETE_FILE = "delete_file"
    CREATE_FOLDER = "create_folder"
    CREATE_FILE = "create_file"
    SHARE_FILE = "share_file"

    # These are writing based actions relating to files and folders

    SAVE_ATTACHMENT = "save_attachment"
    ASK_CLARIFICATION = "ask_clarification"
    DONE = "done"

    # These are some miscellaneous actions relating to attachments, judgement and finishing a task

REQUIRED_PARAMS: dict[ActionType, list] = {
    ActionType.SEARCH_EMAIL:     ["query"],
    ActionType.OPEN_EMAIL:       ["email_id"],
    ActionType.SEND_EMAIL:       ["to", "subject", "body"],
    ActionType.REPLY_EMAIL:      ["email_id", "body"],
    ActionType.FORWARD_EMAIL:    ["email_id", "to"],
    ActionType.ARCHIVE_EMAIL:    ["email_id"],
    ActionType.SEARCH_DRIVE:     ["query"],
    ActionType.OPEN_FILE:        ["file_id"],
    ActionType.MOVE_FILE:        ["file_id", "dest_path"],
    ActionType.COPY_FILE:        ["file_id", "dest_path"],
    ActionType.RENAME_FILE:      ["file_id", "new_name"],
    ActionType.DELETE_FILE:      ["file_id"],
    ActionType.CREATE_FOLDER:    ["path"],
    ActionType.CREATE_FILE:      ["name", "content", "folder_path"],
    ActionType.SHARE_FILE:       ["file_id", "user"],
    ActionType.SAVE_ATTACHMENT:  ["email_id", "attachment_id", "dest_path"],
    ActionType.ASK_CLARIFICATION:["question"],
    ActionType.DONE:             [],
}

 # This is the required parameters datatype that is a dictionary of all the operations and what parameters they should take

def validate_action(action: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(action, dict) or "type" not in action:
        return False, "Action must be a dict with 'type' field."
    try:
        action_type = ActionType(action["type"])
    except ValueError:
        return False, f"Unknown action type: '{action['type']}'"
    params = action.get("params", {})
    missing = [r for r in REQUIRED_PARAMS[action_type] if r not in params]
    if missing:
        return False, f"Missing params for {action_type.value}: {missing}"
    return True, ""

# This is one of the main functions for the operations because it requires us to check each part of the action to see if it is valid.
# It takes a dict for the action that looks like {"type": "move_file", "params": {"file_id": "file_003", "dest_path": "/Finance"}} and then
# returns a boolean about whether the action was valid and a string describing the reason it may not be valid. We first check if the action
# is even a dictionary or if 'type' is in the dictionary because if not then the action is already invalid. Then we make sure the actiontype
# is a valid action type by comparing it to the class of actions we built earlier. Then we define params to be the parameters of the action
# and add any missing parameters to a list, if the list is empty then the action is good to go, if not then we return false and why.

def execute_action(action: dict[str, Any], world: World) -> tuple[str, float, bool]:
    valid, message = validate_action(action)
    if not valid:
        return f"INVALID ACTION: {message}", invalid_action_penalty + step_cost, False

    action_type = ActionType(action["type"])
    params = action.get("params", {})

    lookup = {
        ActionType.SEARCH_EMAIL:      search_email,
        ActionType.OPEN_EMAIL:        open_email,
        ActionType.SEND_EMAIL:        send_email,
        ActionType.REPLY_EMAIL:       reply_email,
        ActionType.FORWARD_EMAIL:     forward_email,
        ActionType.ARCHIVE_EMAIL:     archive_email,
        ActionType.SEARCH_DRIVE:      search_drive,
        ActionType.OPEN_FILE:         open_file,
        ActionType.MOVE_FILE:         move_file,
        ActionType.COPY_FILE:         copy_file,
        ActionType.RENAME_FILE:       rename_file,
        ActionType.DELETE_FILE:       delete_file,
        ActionType.CREATE_FOLDER:     create_folder,
        ActionType.CREATE_FILE:       create_file,
        ActionType.SHARE_FILE:        share_file,
        ActionType.SAVE_ATTACHMENT:   save_attachment,
        ActionType.ASK_CLARIFICATION: ask_clarification,
        ActionType.DONE:              done,
    }

    try:
        result, delta, end = lookup[action_type](params, world)
    except Exception as exc:
        return f"ERROR executing {action_type.value}: {exc}", invalid_action_penalty + step_cost, False

    return result, delta + step_cost, end

# This is the main function of the operations file that lets an action execute. It first calls validate_action to 
# make sure that this is a valid action and returns a message if not. The function takes the action dictionary and
# the World dataclass we defined in state.py and instantiates the action_type and params. Then we build the lookup
# dictionary which makes it easier to see what action we are executing as opposed to a bunch of if statements checking
# which action it is. Then lastly we define result, delta, end to be the output of the action we execute and return
# an error message, the invalid aciton penalty + step cost, and False to show that the action did not work. 

def search_email(params, state):
    ids = state.search_emails(params["query"])
    if not ids:
        return f"No emails found for '{params['query']}'.", 0.0, False
    shown = ids[:10]
    note = f" (showing first 10 of {len(ids)})" if len(ids) > 10 else ""
    lines = []
    for eid in shown:
        e = state.get_email(eid)
        lines.append(f"  {eid}: from={e.sender} | {e.subject}")
    return f"Found {len(ids)} email(s){note}:\n" + "\n".join(lines), 0.0, False

# This is the search_email operation. First we define ids to be the list of emails containing the key word of the query parameter then
# we say if ids does not exist return a message and if it does exist, we define shown which is the top 10 emails found and note which
# indicates that we are showing the first 10 if there are more than that. Then we go through each email id in shown and append a string 
# that contains the email id, the sender and the subject. Then we finally return that we found these emails.

def open_email(params, state):
    e = state.get_email(params["email_id"])
    if e is None:
        return f"Email '{params['email_id']}' not found.", invalid_action_penalty, False
    e.read = True
    atts = ", ".join(f"{a.attachment_id}:{a.name}" for a in e.attachments) or "none"
    return (f"EMAIL {e.email_id}\nFrom: {e.sender}\nSubject: {e.subject}\n"
            f"Folder: {e.folder}\nAttachments: {atts}\n\n{e.body}"), 0.0, False

# This is the open_email operation. First we define e to be the email id of the one the agent wishes to open. If it is none then return 
# an email not found message and invalid aciton penalty, if it does exist though then you define atts to be the attachments of the email
# and then return a string containing all the info of the email.

def send_email(params, state):
    attach_id = params.get("attachment_file_id") 
    attachments = []
    if attach_id is not None:
        f = state.get_file(attach_id)
        if f is None:
            return f"Cannot send: file '{attach_id}' not found.", invalid_action_penalty, False
        attachments.append(Attachment(state.next_id("att"), f.name, f.content))

    to = params["to"]
    recipients = [to] if isinstance(to, str) else to

    eid = state.next_id("email")
    state.emails[eid] = Email(
        email_id=eid,
        sender="me@company.com",          
        subject=params["subject"],
        body=params["body"],
        thread_id=eid,                   
        recipients=recipients,
        folder="Sent",                    
        attachments=attachments,
    )
    return f"Email sent to {', '.join(recipients)} | id={eid}", 0.0, False

# This is the send_email operation. First we define any attachments by taking the optional attachment file id parameter and making a new list 
# of attachments. If there is an attachment id, we get the file relating to that id and append it to the list of attachments. Then we define 
# reciptents to either be a list if the agent sends to multiple people or just the single person being sent to. Then we just define a new email
# id and add a new Email to the emails list.

def reply_email(params, state):
    e = state.get_email(params["email_id"])
    if e is None:
        return f"Email '{params['email_id']}' not found.", invalid_action_penalty, False
    reply_id = state.next_id("email")
    state.emails[reply_id] = Email(
        email_id=reply_id,
        sender="me@company.com",
        subject=e.subject if e.subject.startswith("Re:") else f"Re: {e.subject}",
        body=params["body"],
        thread_id=e.thread_id,          
        recipients=[e.sender],        
        folder="Sent",
    )
    return f"Replied to {e.sender} in thread {e.thread_id} | id={reply_id}", 0.0, False

# This is the reply_email operation. First we define e to be the email itself based on the email id parameter. Then we define
# the reply_id to be a new id and then add that new reply Email to the emails list.

def forward_email(params, state):
    e = state.get_email(params["email_id"])
    if e is None:
        return f"Email '{params['email_id']}' not found.", invalid_action_penalty, False
    to = params["to"]
    recipients = [to] if isinstance(to, str) else to
    fwd_id = state.next_id("email")
    state.emails[fwd_id] = Email(
        email_id=fwd_id,
        sender="me@company.com",
        subject=e.subject if e.subject.startswith("Fwd:") else f"Fwd: {e.subject}",
        body=f"{params.get('note', '')}\n\n--- Forwarded ---\nFrom: {e.sender}\n\n{e.body}",
        thread_id=fwd_id,               # forward STARTS a new thread (new convo with a new person)
        recipients=recipients,
        folder="Sent",
        attachments=list(e.attachments),  # carry the original's attachments along
    )
    return f"Forwarded {e.email_id} to {', '.join(recipients)} | id={fwd_id}", 0.0, False

# This is the forward_email operation. To start we define e to be the email based on the email id parameter. Then we define reciptents
# the same way we did in the send_email operation and make a new email that contains the new information as well as the forwarded emails
# info. Fowarded emails also have a new email id so we define that as fwd_id

def archive_email(params, state):
    e = state.get_email(params["email_id"])
    if e is None:
        return f"Email '{params['email_id']}' not found.", invalid_action_penalty, False
    e.folder = "Archive"
    return f"Archived {e.email_id}.", 0.0, False

# This is the archive_email operation. We define e to be the email id and then if it doesnt exist we return a message. If it does exist we 
# change the folder of the email to archive and return a message indicating this change.

def search_drive(params, state):
    ids = state.search_files(params["query"])
    if not ids:
        return f"No files found for '{params['query']}'.", 0.0, False
    shown = ids[:10]
    note = f" (showing first 10 of {len(ids)})" if len(ids) > 10 else ""
    lines = []
    for fid in shown:
        f = state.get_file(fid)
        lines.append(f"  {fid}: {f.folder_path}/{f.name}")
    return f"Found {len(ids)} file(s){note}:\n" + "\n".join(lines), 0.0, False

# This is the search_drive operation. First we define ids to be the query of the files the agent wishes to find. If it does not exist
# we return a not found message and if it does exist we define lines to be a list of the top 10 files with their folder id, path and name.
# Has the same logic as search_email when it comes to showing the top 10 items because this encourages the agent to have more refined searches

def create_file(params, state):
    if not state.folder_exists(params["folder_path"]):
        return f"Folder '{params['folder_path']}' does not exist.", invalid_action_penalty, False
    new_id = state.next_id("file")
    state.files[new_id] = DriveFile(
        file_id=new_id, name=params["name"],
        content=params["content"], folder_path=params["folder_path"],
    )
    return f"Created file {new_id} in {params['folder_path']}.", 0.0, False

# This is the create_file operation. First we check if the folder path even exists and if it does not we return a not found message with a penalty.
# Then we define a new file id, and then just creaet a new DriveFile object with the corresponding infromation and add it to the files dict in 
# the world class.

def open_file(params, state):
    f = state.get_file(params["file_id"])
    if f is None:
        return f"File '{params['file_id']}' not found.", invalid_action_penalty, False
    return f"FILE {f.file_id}\nName: {f.name}\nLocation: {f.folder_path}\n\n{f.content}", 0.0, False

# This is the open_file operation. We define f to be the file id of the request and then if it does not exist we return a not found and invalid penality.
# If it does exist we return the file id, the name and the locaiton as well as the content of the file.

def move_file(params, state):
    f = state.get_file(params["file_id"])
    if f is None:
        return f"File '{params['file_id']}' not found.", invalid_action_penalty, False
    if not state.folder_exists(params["dest_path"]):
        return f"Folder '{params['dest_path']}' does not exist.", invalid_action_penalty, False
    old = f.folder_path
    f.folder_path = params["dest_path"]
    return f"Moved {f.file_id} from {old} to {params['dest_path']}.", 0.0, False

# This is the move_file operation. We define f to be the file id. If f does not exist return a not found and penalty. Next we check if the destination path 
# from the parameters exists, if not we return a not found and penalty. Then if both exist we define old to be the current folder path and then we change 
# the folder path of the file to the desired one and print a message showing it has been moved from one folder to the other.

def copy_file(params, state):
    f = state.get_file(params["file_id"])
    if f is None:
        return f"File '{params['file_id']}' not found.", invalid_action_penalty, False
    if not state.folder_exists(params["dest_path"]):
        return f"Folder '{params['dest_path']}' does not exist.", invalid_action_penalty, False
    new_id = state.next_id("file")
    state.files[new_id] = DriveFile(
        file_id=new_id,                 # the copy gets a NEW id — it's a distinct file
        name=f.name,
        content=f.content,
        folder_path=params["dest_path"],
    )
    return f"Copied {f.file_id} to {params['dest_path']} as {new_id}.", 0.0, False

# This is thte copy_file operation. We first define f to be the file id and if it does not exist or the folder the file is in does not exist
# then we return a not found error and a penalty. Then we create a new file id by doing next_id and add this new coyp of a file to the files
# dictionary.

def rename_file(params, state):
    f = state.get_file(params["file_id"])
    if f is None:
        return f"File '{params['file_id']}' not found.", invalid_action_penalty, False
    old = f.name
    f.name = params["new_name"]
    return f"Renamed '{old}' to '{params['new_name']}'.", 0.0, False

# This is the rename_file operation. We define f to be the file id of the params and return message if it doesnt exist. Then we define old to be the current name
# and change the name of the file to the desired one and return a message indicating this action.

def delete_file(params, state):
    fid = params["file_id"]
    if fid not in state.files:
        return f"File '{fid}' not found.", invalid_action_penalty, False
    name = state.files[fid].name
    del state.files[fid]
    return f"Deleted file '{name}' ({fid}).", 0.0, False

# This is the delete_file operation. We first define the file id from the paraemters. Then if it is not a valid file we return the corresponding message
# and then we remove the file based on the idea from the state.files dictionary.

def create_folder(params, state):
    path = params["path"]
    if path in state.folders:
        return f"Folder '{path}' already exists.", 0.0, False
    state.folders.add(path)
    return f"Created folder '{path}'.", 0.0, False

# This is the create_folder operation. We define the path from the parameters and if that path is in the folders set already we return that it already
# exists and if not we add the new path to the folders set.

def share_file(params, state):
    f = state.get_file(params["file_id"])
    if f is None:
        return f"File '{params['file_id']}' not found.", invalid_action_penalty, False
    user = params["user"]
    if user not in f.shared_with:
        f.shared_with.append(user)
    else:
        return f"{f.file_id} already shared with {user}.", 0.0, False
    return f"Shared {f.file_id} with {user}.", 0.0, False

# This is the share_file operation. First we define f to be the file based on the file id parameter and if it does not exist output a message. 
# then we define user to be the user parameter and if user is not in the files shared with variable already we add it and if it is we return a 
# message indicating the file is already shared.

def save_attachment(params, state):
    e = state.get_email(params["email_id"])
    if e is None:
        return f"Email '{params['email_id']}' not found.", invalid_action_penalty, False
    att = next((a for a in e.attachments if a.attachment_id == params["attachment_id"]), None)
    if att is None:
        return f"Attachment '{params['attachment_id']}' not found in {e.email_id}.", invalid_action_penalty, False
    if not state.folder_exists(params["dest_path"]):
        return f"Folder '{params['dest_path']}' does not exist.", invalid_action_penalty, False
    new_id = state.next_id("file")
    state.files[new_id] = DriveFile(
        file_id=new_id,
        name=att.name,
        content=att.content,           
        folder_path=params["dest_path"],
    )
    return f"Saved attachment '{att.name}' to {params['dest_path']} as {new_id}.", 0.0, False

# This is the save_attachment operation. We define first the email that we are accessing using the parameters. Then if e does not exist
# we return the not found message. Then we define att to be the attachment that matches the specific attachment id from the params. If it is
# none we return a penalty and not found message. Then if the folder that the agent wants to put the file in does not exist we do the same.
# Then if everything exists we make a new file id, and add a new file to the state files dict holding the attachment info.

def ask_clarification(params, state):
    state.clarifications.append(params["question"])  
    return f"Asked: {params['question']}", 0.0, False

# This is the ask_clarification operation which adds the question to the state clarifications list for verifying purposes and thats itt

def done(params, state):
    state.done_answer = params.get("answer")        
    return f"Episode ended. answer={state.done_answer}", 0.0, True

# This is the done operation which the agent executes after its done with its entire task and the answer it may provide after finishing the task
# is then stored in the state.done_answer str variable.



