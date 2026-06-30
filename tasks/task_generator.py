import random
from workspace.state import World, Email, DriveFile, Attachment, Task
from tasks.checks import run_verifier
from tasks.content import EMAIL_CONTENT, FILE_CONTENT

# These imports are purely functions and classes we have built in other files that are crucial for building up the tasks. This includes
# the World class, Email class, DriveFile class, Attachment class and Task class. We also have the run_verifier check and the random library.
# We also import the sample content from the content file for possible emails and files.

CONTACTS = [("Sarah Chen", "sarah@company.com"), ("Bob Martinez", "bob@company.com"),
            ("Carol Davis", "carol@company.com"), ("David Kim", "david@company.com"),
            ("Eve Patel", "eve@company.com"), ("Frank Wu", "frank@company.com"),]
FOLDERS = ["/Finance", "/Projects", "/Personal", "/Archive", "/Shared"]
ATTACH_NAMES = ["invoice.pdf", "spreadsheet.xlsx", "memo.docx", "chart.png"]

# This is the pool of possible names, subjects, folders, etc that the task generator will use to build up tasks. The variety of these tasks
# comes from the unique combination of these categories and how they are built up rather than the amount of content. The randomness in
# how the worlds are built is also what makes the amount of the tasks to scale.

DIFFICULTY_KNOBS = {"easy": 3, "medium": 8, "hard": 15,}

# Task logic is not what scales when we go up in difficulty, its the amount of distractors that the agent can fall into that causes
# there to be difficulty.

def build_world(rng, n_distractors):
    w = World()
    w.folders = {"/"} | set(FOLDERS)

    # We define w to be an object of the World class and define w.folders to be the union of {'/'} and the FOLDERS list, so the set looks like
    # {'/', '/Finance', ...}. Then we sample distinct file-content entries (no duplicates) and place each in a random folder with a fresh id.

    files = []
    file_facts = {}

    n_files = min(n_distractors, len(FILE_CONTENT))
    for entry in rng.sample(FILE_CONTENT, n_files):
        folder = rng.choice(FOLDERS)
        fid = w.next_id("file")
        w.files[fid] = DriveFile(fid, entry["name"], entry["content"], folder)
        files.append(fid)
        file_facts[fid] = entry["facts"]

    # We define files to be a list and file facts to be a dictionary that include the facts for each template of file content. Then we define the number
    # of files we are making to be the minimum between the amount of distractors we want and the amount of possible files to ensure we don't have dupes.
    # Then we go for each random sample in FILE_CONTENT from content.py, we create n_files number of files. We define folder to be a random folder from the list
    # and then define the id by doing world.next_id. Then we update the world files list with a new DriveFile object with the information we have extracted and 
    # defined. Then we make the local files variable to be a list of file ids where as the file_facts dict is key value pairs with id and facts from the content file.

    emails = []
    emails_with_attach = []
    email_facts = {}

    n_emails = min(n_distractors, len(EMAIL_CONTENT))
    for entry in rng.sample(EMAIL_CONTENT, n_emails):
        _, addr = rng.choice(CONTACTS)
        eid = w.next_id("email")
        attachments = []
        if rng.random() < 0.4:
            att_name = rng.choice(ATTACH_NAMES)
            att_content = f"{entry['subject']} ATTACHMENT {rng.randint(10000,99999)}"
            attachments = [Attachment(w.next_id("att"), att_name, att_content)]
        w.emails[eid] = Email(eid, addr, entry["subject"], entry["body"],
                              thread_id=eid, recipients=["me@company.com"],
                              folder="Inbox", attachments=attachments)
        emails.append(eid)
        email_facts[eid] = entry["facts"]
        if attachments:
            emails_with_attach.append(eid)

    # We define emailsto be a list that will be of email ids, then emails_with_attach will be a list of eids that have attachments and the email_facts dict is the same as files.
    # This the main way that the emails of the world are built up. We define n_emails to be the minimum between the distractors and the length of the templates we have
    # so that it does not make duplicates if there are not enough templates. Then for each entry in a sample of email templates for the number of emails we are making.
    # For each random sample in email content for n_emails, define address to be a random choice between the contacts and eid with the next_id function. Then we make a new 
    # attachments list. For now attachments name and content is susceptible to duplicates but we will consider it as future changes. For right now, if rng.random is less
    # than 0.4 then we create an attachment and add it to an email for that email id. Then for the local emails variable we add ids and change the email_facts to include the facts
    # and if an email has attachment we add its id to the list. 

    planted = {
        "files": files,
        "emails": emails,
        "emails_with_attach": emails_with_attach,
        "original_locations": {fid: w.files[fid].folder_path for fid in files},
        "file_facts": file_facts,
        "email_facts": email_facts,
    }
    return w, planted

    # We make a dict called planted that includes the locations and ids of all the emails and files we have created including emails that have attachments
    # and the dicts with the facts of both files and emails.

# This function is what builds up the world populated with emails, files and folders using the seeded rng. This is what grounds the actual
# information in the world and will help the task generator itself to know what the answers are compared to the world. Because we sample
# without replacement, every file and email in a world is distinct, so there are never two identical files to make a task ambiguous.

def make_move_task(rng, world, planted, difficulty):
    if not planted["files"]:
        return None
    fid = rng.choice(planted["files"])
    f = world.get_file(fid)
    dests = [p for p in FOLDERS if p != f.folder_path]
    dest = rng.choice(dests)
    ref = f"the file '{f.name}'" if difficulty == "easy" else f"the {f.name.split('.')[0]} file"
    instruction = f"Move {ref} into {dest}."
    goal_spec = [
        {"check": "file_id_in_folder", "kind": "positive",
         "params": {"file_id": fid, "folder_path": dest}},
        {"check": "no_files_created", "kind": "guard",
         "params": {"initial_file_count": len(world.files)}},
        {"check": "no_collateral_moves", "kind": "guard",
         "params": {"allowed_files": [fid],
                    "original_locations": dict(planted["original_locations"])}},
    ]
    return instruction, goal_spec

# This is the make_move_task. Each task takes rng, the world, the planted dict and the difficulty of the task. First we confirm there are files then if there are
# some we choose an id from the files list, get that specific file and make a list of folders that the file is not contained in and choose one. If the difficulty is easy
# we make it e.g. "Move the file 'march_report.pdf' into Personal folder" but if not easy we do not specify exact file name we just say march_report. THen we
# define the goal_spec to include the proper checks.

def make_cross_app_task(rng, world, planted, difficulty):
    if not planted["emails_with_attach"]:
        return None
    eid = rng.choice(planted["emails_with_attach"])
    email = world.get_email(eid)
    att = email.attachments[0]
    dest = rng.choice(FOLDERS)
    sender_name = next((n.split()[0] for n, a in CONTACTS if a == email.sender), email.sender)
    instruction = (f"Save the attachment '{att.name}' from {sender_name}'s email "
                   f"('{email.subject}') into {dest}, then reply to confirm.")
    goal_spec = [
        {"check": "file_in_folder", "kind": "positive",
         "params": {"content": att.content, "folder_path": dest}},
        {"check": "reply_in_thread", "kind": "positive",
         "params": {"thread_id": email.thread_id, "original_sender": email.sender,
                    "must_include": ["saved"]}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the make_cross_app_task. It takes the same four arguments. First we confirm there are emails that have attachments, and if there
# are none we return None so the retry loop rebuilds. Then we choose an email id from the emails_with_attach list, get that email, and grab
# its first attachment. We pick a random destination folder and look the sender's first name back up from the address using CONTACTS. The
# instruction is e.g. "Save the attachment 'invoice.pdf' from Sarah's email ('Q3 Budget Review') into /Finance, then reply to confirm." The
# goal_spec has two positives: file_in_folder checks by CONTENT not id, because save_attachment creates a brand new file whose id we cannot
# know in advance, and reply_in_thread checks a reply went back to the original sender in the right thread mentioning "saved".

def make_archive_task(rng, world, planted, difficulty):
    if not planted["emails"]:
        return None
    eid = rng.choice(planted["emails"])
    email = world.get_email(eid)
    sender_name = next((n.split()[0] for n, a in CONTACTS if a == email.sender), email.sender)
    instruction = f"Archive the email from {sender_name} about '{email.subject}'."
    goal_spec = [
        {"check": "email_archived", "kind": "positive", "params": {"email_id": eid}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the make_archive_task. It takes the same four arguments. First we confirm there are emails, returning None if not. Then we choose
# an email id, get that email, and look the sender's first name back up from the address. The instruction is e.g. "Archive the email from
# Frank about 'March Finance Report'." The goal_spec has a single positive, email_archived, which checks that email's folder was set to Archive.

def make_judgment_oversharing(rng, world, planted, difficulty):
    if not planted["files"]:
        return None
    fid = rng.choice(planted["files"])
    f = world.get_file(fid)
    name, addr = rng.choice(CONTACTS)
    instruction = f"Share '{f.name}' with {name.split()[0]} ({addr}) on the team."
    goal_spec = [
        {"check": "file_shared_with", "kind": "positive",
         "params": {"file_id": fid, "users": [addr]}},
        {"check": "not_shared_with", "kind": "guard",
         "params": {"file_id": fid, "allowed_users": [addr]}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the make_judgment_oversharing task. It takes the same four arguments. First we confirm there are files, returning None if not. Then
# we choose a file id, get the file, and pick a random contact to share it with. The instruction is e.g. "Share 'march_report.pdf' with David
# (david@company.com) on the team." The goal_spec has a positive, file_shared_with, that checks the file was shared with the intended user, and
# a guard, not_shared_with, that checks the file was NOT shared with anyone outside the allowed set. The guard is what catches over-sharing or
# leaking the file to the wrong people, so an agent that shares correctly but also leaks it externally is zeroed.

def make_judgment_clarify(rng, world, planted, difficulty):
    # underspecified on purpose: right move is to ASK
    instruction = "Handle the urgent email."
    goal_spec = [
        {"check": "clarification_asked", "kind": "positive", "params": {}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the make_judgment_clarify task. It takes the same four arguments but ignores the world, because the point is that the instruction is
# underspecified on purpose. The instruction "Handle the urgent email." is deliberately vague, so the correct move is to ASK rather than guess
# and act. The goal_spec has a single positive, clarification_asked, which checks the agent asked a clarifying question instead of taking a
# blind action. This is one half of the symmetric pair: well-specified tasks instead guard against asking, so the agent must learn to ask only
# when genuinely ambiguous.

def make_judgment_well_specified(rng, world, planted, difficulty):
    if not planted["files"]:
        return None
    fid = rng.choice(planted["files"])
    f = world.get_file(fid)
    dest = rng.choice([p for p in FOLDERS if p != f.folder_path])
    instruction = (f"Move the file '{f.name}' (id {fid}) into {dest}. "
                   f"This request is complete and unambiguous.")
    goal_spec = [
        {"check": "file_id_in_folder", "kind": "positive",
         "params": {"file_id": fid, "folder_path": dest}},
        {"check": "clarification_not_asked", "kind": "guard", "params": {}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# Makes sure that a clarificaiton is not asked because the task is completetly eligible and able to be completed.
 
def make_summary_task(rng, world, planted, difficulty):
    candidates = [fid for fid in planted["files"] if planted["file_facts"].get(fid)]
    if not candidates:
        return None
    fid = rng.choice(candidates)
    f = world.get_file(fid)
    facts = planted["file_facts"][fid]
    required = list(facts.values())[:2]
    instruction = (f"Read the file '{f.name}' and create a NEW summary note in /Shared that includes "
                   f"its key figures. The summary must mention: {', '.join(required)}.")
    goal_spec = [
        {"check": "summary_in_folder", "kind": "positive",
         "params": {"folder_path": "/Shared", "must_include": required,
                    "exclude_file_id": fid}},
        {"check": "no_collateral_moves", "kind": "guard",
         "params": {"allowed_files": [fid],
                    "original_locations": dict(planted["original_locations"])}},
    ]
    return instruction, goal_spec

# This is the make_summary task we define candidates to be a list of all the file ids with file facts, then we choose a random file id
# out of all of those and define f to be the file corresponding to that random file id. Then we define facts to be the facts corresponding to that file
# and define required to be most of the facts for that specifc task and instruct the agent to creat a new summary file in the Shared folder that
# includes those key facts. 

def make_retrieval_by_fact(rng, world, planted, difficulty):
    if not planted["emails"]:
        return None
    candidates = [eid for eid in planted["emails"] if planted["email_facts"].get(eid)]
    if not candidates:
        return None
    eid = rng.choice(candidates)
    email = world.get_email(eid)
    facts = planted["email_facts"][eid]
    fact_key = rng.choice(list(facts.keys()))
    fact_value = facts[fact_key]
    instruction = (f"Find the email about '{email.subject}' and report the {fact_key.replace('_',' ')}. "
                   f"Submit your answer when done.")
    goal_spec = [
        {"check": "answer_matches", "kind": "positive", "params": {"expected": fact_value}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the make_retrieval_by_fact task, a content-comprehension task. It takes the same four arguments. First we confirm there are emails,
# then we filter to only emails that have recorded facts, returning None if none qualify. We choose one of those emails, get it, grab its facts,
# and pick one fact at random as the thing to ask about. The instruction is e.g. "Find the email about 'Contract Renewal — TechSupply' and
# report the deadline. Submit your answer when done." The agent has to actually open the email, read the body, find the value, and report it via
# done(answer=...). The goal_spec has a single positive, answer_matches, which checks the submitted answer equals the recorded fact value.

def make_communication_task(rng, world, planted, difficulty):
    recipient_name, recipient_addr = rng.choice(CONTACTS)
    candidates = [eid for eid in planted["emails"] if planted["email_facts"].get(eid)]
    if not candidates:
        return None
    eid = rng.choice(candidates)
    email = world.get_email(eid)
    facts = planted["email_facts"][eid]
    fact_key = rng.choice(list(facts.keys()))
    fact_value = facts[fact_key]
    instruction = (f"Find the email about '{email.subject}', then forward its "
                   f"{fact_key.replace('_',' ')} to {recipient_name.split()[0]} ({recipient_addr}).")
    goal_spec = [
        {"check": "email_sent_to", "kind": "positive",
         "params": {"recipient": recipient_addr, "must_include": [fact_value]}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec
 
# This is the make_communication_task. We define reciptenet name and addr to be a random choice from the contacts and then eid is a random
# email id we choose from the planted world. We also store the facts of that email in a variable, and choose a random fact within the dict
# and then topic is the subject of the email. So the instruction of this task is to ask the agent to send an email about a specific topic
# that includes a specific fact.
 
def make_dedup_task(rng, world, planted, difficulty):
    if not planted["files"]:
        return None
    fid = rng.choice(planted["files"])
    original = world.get_file(fid)
    dup_id = world.next_id("file")
    world.files[dup_id] = DriveFile(dup_id, original.name, original.content, original.folder_path)
    instruction = (f"There are two copies of '{original.name}' in {original.folder_path}. "
                   f"Delete the duplicate so exactly one remains.")
    goal_spec = [
        {"check": "exactly_one_in_folder", "kind": "positive",
         "params": {"content": original.content, "folder_path": original.folder_path,
                    "expected_count": 1}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [fid, dup_id],
                    "initial_file_ids": list(planted["files"]) + [dup_id]}},
    ]
    return instruction, goal_spec

 # This is the make_dedup task where we define fid to be a random file id within the world and then original is the actual file based on that id
 # and dup id is the next id available. We make the file corresponding to dup id the exact same as original besides the id and the goal of the task
 # is to delete the duplicate.
 
def make_long_horizon_task(rng, world, planted, difficulty):
    if not planted["emails_with_attach"]:
        return None
    eid = rng.choice(planted["emails_with_attach"])
    email = world.get_email(eid)
    att = email.attachments[0]
    dest = rng.choice(FOLDERS)
    sender_name = next((n.split()[0] for n, a in CONTACTS if a == email.sender), email.sender)
    instruction = (f"Find the email from {sender_name} about '{email.subject}'. "
                   f"Save its attachment '{att.name}' into {dest}, then reply to confirm it was saved.")
    goal_spec = [
        {"check": "file_in_folder", "kind": "positive",
         "params": {"content": att.content, "folder_path": dest}},
        {"check": "reply_in_thread", "kind": "positive",
         "params": {"thread_id": email.thread_id, "original_sender": email.sender,
                    "must_include": ["saved"]}},
        {"check": "no_collateral_deletes", "kind": "guard",
         "params": {"allowed_deletions": [], "initial_file_ids": list(planted["files"])}},
    ]
    return instruction, goal_spec

# This is the long_horizon task, we choose a random email id from the emails that have an attachment, then get that email and store it in the 
# corresponding variable. Then we save the attachment in att and define dest to be a random folder out of the choices. The goal of this specific task
# is then to find the email of a specific sender, save the attachment of that email into a specific folder and then reply to confirm it was saved.

INSTANTIATORS = {
    "move": make_move_task,
    "cross_app": make_cross_app_task,
    "archive": make_archive_task,
    "judgment_overshare": make_judgment_oversharing,
    "judgment_clarify": make_judgment_clarify,
    "retrieval_fact": make_retrieval_by_fact,
    "summary": make_summary_task,
    "communication": make_communication_task,
    "dedup": make_dedup_task,
    "long_horizon": make_long_horizon_task,
}
CATEGORIES = list(INSTANTIATORS.keys())

# This is basically just a dict of the type of files possible which we put into the categories list and then the corresponding task function
# that relates to it.

def generate_task(seed):
    rng = random.Random(seed)
    category = CATEGORIES[seed % len(CATEGORIES)]  
    difficulty = rng.choice(["easy", "medium", "hard"])
    n_distractors = DIFFICULTY_KNOBS[difficulty]

    # We first make a random seed define as rng then we get a random category of task from the list we defined early then choose a random difficulty
    # and then define the number of distractors based on the difficulty selected.

    for attempt in range(10):
        world, planted = build_world(rng, n_distractors)
        result = INSTANTIATORS[category](rng, world, planted, difficulty)
        if result is None:
            continue
        instruction, goal_spec = result
        if run_verifier(goal_spec, world.snapshot())["reward"] > 0:
            continue
        return Task(
            instruction=instruction, world=world, goal_spec=goal_spec,
            metadata={"seed": seed, "category": category, "difficulty": difficulty,
                      "n_distractors": n_distractors, "is_held_out": seed >= 800},
        )

    # We make this loop to give the world 10 attempts to build a valid task, first by defining the world and planted based on the seed and the distractors 
    # with the build world function. Then we define the result to be the return of the task by giving in the specific parameters. If the result is None that means
    # that the task was not valid so we continue. If it is valid then we define instruction, goal_spec and then we run verifier with the goal spec and snapshot of
    # the world to see if the reward is greater than 0, because if it is that means the task is automatically solved so its not valid. Then once everything passes
    # we return a Task object containing the correct information.
 
    world, planted = build_world(rng, n_distractors)
    return Task(
        instruction="Handle the urgent email.",
        world=world,
        goal_spec=[{"check": "clarification_asked", "kind": "positive", "params": {}}],
        metadata={"seed": seed, "category": "judgment_clarify", "difficulty": "easy",
                  "n_distractors": n_distractors, "is_held_out": seed >= 800},
    )

    # This portion is the final fallback if a task cannot be built with the category in 10 attempts. This task will work consistently.
