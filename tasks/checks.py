from workspace.state import World

def file_in_folder(world, params):
    return any(f.content == params["content"] and f.folder_path == params["folder_path"] 
               for f in world.files.values())

# This is the first file in folder function that can be used to check if a specific file was saved by checking the content of the file.
# It essentially loops through all files and compares it to the content of the file that was required to be saved. The params are from 
# the goal_spec.

def file_id_in_folder(world, params):
    f = world.get_file(params["file_id"])
    return f is not None and f.folder_path == params["folder_path"]

# This is the other file in folder function that checks based on id. This one checks if the file id of the agents move even exists and if 
# and if it has the proper folder path. This one is more important for actions where the agent moves files around as opposed to creating new 
# files like in the other instance.

def no_collateral_moves(world, params):
    allowed = set(params["allowed_files"])
    for file_id, original_folder in params["original_locations"].items():
        if file_id in allowed:
            continue
        f = world.get_file(file_id)
        if f is None or f.folder_path != original_folder:
            return False
    return True

# This function is made to make sure that whatever action the agent takes, there are no collateral moves it made that changed the locations
# of files that did not need to be changed. So we do this by taking the allowed files from the params and then compare each file to see if it was 
# allowed to move and if not we check that it exists and is in the same original folder as it is supposed to be. If it isnt we pass False and if
# all files are in the right place we pass True.

def email_sent_to(world, params):
    return any(
        params["recipient"] in e.recipients
        and e.folder == "Sent"
        and all(c.lower() in e.body.lower() for c in params.get("must_include", []))
        for e in world.emails.values()
    )

# This makes sure that the email that was sent was not only sent to the right person but also make sure the content of the email
# matches what must be in the email that was sent. This is a little bit difficult because it is relying on text and not concrete structure of the world.

def reply_in_thread(world, params):
    return any(
        e.folder == "Sent"
        and e.thread_id == params["thread_id"]
        and params["original_sender"] in e.recipients
        and all(c.lower() in e.body.lower() for c in params.get("must_include", []))
        for e in world.emails.values()
    )

# This function checks to see if there is reply in the correct thread by checking thread_id, content and if the email is in the sent inbox of the email sent is correct. 

def email_not_deleted(world, params):
    return all(email_id in world.emails for email_id in params["email_ids"])

# Just checks if an email is not deleted by checking that the email id is not present in the emails dict

def clarification_asked(world, params):
    return len(world.clarifications) > 0

# Checks if the agent even asked for any clarification by seeing if the length of the world list is greater than 0

def clarification_not_asked(world, params):
    return len(world.clarifications) == 0

# Checks if no clarification was asked at all by seeing if the length of the list is 0

def answer_matches(world, params):
    submitted = (world.done_answer or "").strip().lower()
    return submitted == params["expected"].strip().lower()

# If the task requires an answer to be at the end we are checking that the agents answer corresponds to the correct one

def file_renamed(world, params):
    f = world.get_file(params["file_id"])
    return f is not None and f.name == params["new_name"]

# This function makes sure the file is properly renamed given the tasks instructions

def email_archived(world, params):
    e = world.get_email(params["email_id"])
    return e is not None and e.folder == "Archive"

# This function makes sure that a email is archived if required by the task

def file_deleted(world, params):
    return params["file_id"] not in world.files

# Checks if a file has been properly deleted

def exactly_one_in_folder(world, params):
    count = sum(1 for f in world.files.values()
                if f.folder_path == params["folder_path"] and f.content == params["content"])
    return count == params["expected_count"]

# Makes sure there is the proper amount of files in a specific folder

def file_shared_with(world, params):
    f = world.get_file(params["file_id"])
    return f is not None and all(u in f.shared_with for u in params["users"])

# Makes sure that a file that should be shared has been properly shared with the correct people

def not_shared_with(world, params):
    f = world.get_file(params["file_id"])
    if f is None:
        return False
    return all(u in params["allowed_users"] for u in f.shared_with)

# Makes sure files have not been shared with the wrong people

def file_has_content(world, params):
    target = world.get_file(params["file_id"]) if "file_id" in params else None
    if target is None:
        return False
    return all(fact.lower() in target.content.lower() for fact in params["must_include"])

# Makes sure a newly created has the proper content

def no_files_created(world, params):
    return len(world.files) <= params["initial_file_count"]

# Checks if no new random files have not been created

CHECKS = {
    "file_in_folder":          file_in_folder,
    "file_id_in_folder":       file_id_in_folder,
    "no_collateral_moves":     no_collateral_moves,
    "no_files_created":        no_files_created,
    "email_sent_to":           email_sent_to,
    "reply_in_thread":         reply_in_thread,
    "email_archived":          email_archived,
    "email_not_deleted":       email_not_deleted,
    "file_renamed":            file_renamed,
    "file_deleted":            file_deleted,
    "exactly_one_in_folder":   exactly_one_in_folder,
    "file_shared_with":        file_shared_with,
    "not_shared_with":         not_shared_with,
    "file_has_content":        file_has_content,
    "clarification_asked":     clarification_asked,
    "clarification_not_asked": clarification_not_asked,
    "answer_matches":          answer_matches,
}

# This is just a dict referencing all the checks within this file.

def run_verifier(goal_spec, world):
    results = []
    for entry in goal_spec:
        fn = CHECKS[entry["check"]]
        passed = fn(world, entry["params"])
        kind = entry.get("kind", "positive")   
        results.append({"check": entry["check"], "params": entry["params"],
                        "passed": passed, "kind": kind})

    positives = [r for r in results if r["kind"] == "positive"]
    guards    = [r for r in results if r["kind"] == "guard"]

    if any(not g["passed"] for g in guards):
        reward = 0.0
    elif positives:
        reward = sum(1 for p in positives if p["passed"]) / len(positives)
    else:
        reward = 0.0

    return {"reward": reward, "predicate_results": results,
            "passed": sum(1 for r in results if r["passed"]), "total": len(results)}

# With an example goal_spec looking like this:

"""goal_spec = [
    {
        "check": "file_id_in_folder",
        "kind": "positive",
        "params": {"file_id": "file_001", "folder_path": "/Finance"}
    },
    {
        "check": "no_files_created",
        "kind": "guard",
        "params": {"initial_file_count": 14}
    },
    {
        "check": "no_collateral_moves",
        "kind": "guard",
        "params": {
            "allowed_files": ["file_001"],
            "original_locations": {"file_002": "/Projects", "file_003": "/Personal"}
        }
    },
]"""

# This function takes the goal spec and breaks it down to return information about the completed task like reward and checks passed. It starts by
# instantiating results to be an empty list. Then for each dict (check) in goal_spec, you define fn to be the check itself and then passed to be the return 
# of whatever check you are doing by inputting the specific dicts params. Then we define kind to be the type of check that defaults to positive. Then 
# we put all this info into a list known as result. Then we must count the number of positive types of checks which are using task goals and we also
# count the number of guard checks for anti-gaming. If any of the guards were not passed, the reward is 0 and same with if no checks were passed. If there
# are some passes in the positives list, we make the reward the number of passed divided by the number total checks. Then we finally return a dict
# with the reward, results, number of passed checks and the total number of checks.