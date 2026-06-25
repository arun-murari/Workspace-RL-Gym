from agents.agent import Agent

# Import the rule book from the Agent class where we can rewrite the reset and action functions

class ScriptedOracleBaseline(Agent):
    def reset(self, task):
        self.plan = self._plan_from_goal(task)
        self.step = 0

    # This is the baseline agents reset function that defines the plan of the agent to be the return from the plan_from_goal function according to the task
    # and then we initialize the step count to be 0

    def act(self, observation, task):
        if self.step < len(self.plan):
            action = self.plan[self.step]
            self.step += 1
            return action
        return {"type": "done", "params": {}}

    # This is the act function that defines what move the agent will take as it goes to complete the task. If the step count is less than the current plan
    # you make the current action equal to the current step in the plan, increase step by one and then return action. After that is done you return the done
    # operation to indicate a finished task.

    def _plan_from_goal(self, task):
        plan = []
        world = task.world

        # We define plan to be an empty list and world to be the world define by the task.

        for entry in task.goal_spec:
            check = entry["check"]
            p = entry["params"]

            # For each entry in the goal_spec define check to be the check entry of the each item and p to be the params.

            # Then from this point on, we do a series of if statements to see which check will be required of the task which allows the 
            # agent to choose the right actions and set the right parameters. 

            if check == "file_id_in_folder":         
                plan.append({"type": "move_file",
                             "params": {"file_id": p["file_id"], 
                                        "dest_path": p["folder_path"]}})

            elif check == "file_in_folder":           
                for e in world.emails.values():
                    for att in e.attachments:
                        if att.content == p["content"]:
                            plan.append({"type": "save_attachment",
                                         "params": {"email_id": e.email_id,
                                                    "attachment_id": att.attachment_id,
                                                    "dest_path": p["folder_path"]}})
                            break

            elif check == "reply_in_thread":         
                for e in world.emails.values():
                    if e.thread_id == p["thread_id"]:
                        body = " ".join(p.get("must_include", ["confirmed"]))
                        plan.append({"type": "reply_email",
                                     "params": {"email_id": e.email_id, "body": body + " saved"}})
                        break

            elif check == "email_archived":          
                plan.append({"type": "archive_email", "params": {"email_id": p["email_id"]}})

            elif check == "file_renamed":
                plan.append({"type": "rename_file",
                             "params": {"file_id": p["file_id"], "new_name": p["new_name"]}})

            elif check == "file_deleted":
                plan.append({"type": "delete_file", "params": {"file_id": p["file_id"]}})

            elif check == "file_shared_with":        
                for u in p["users"]:
                    plan.append({"type": "share_file",
                                 "params": {"file_id": p["file_id"], "user": u}})

            elif check == "clarification_asked":     
                plan.append({"type": "ask_clarification",
                             "params": {"question": "Which item do you mean?"}})

            elif check == "answer_matches":         
                plan.append({"type": "done", "params": {"answer": p["expected"]}})

        return plan

# We return plan at the end. The general idea of this file is that we are building a scripted oracle agent where the agent
# can essentially see the right answers by inspecting the goal spec. You might think that this defeats the purpose but the 
# point of this is to show that the environment is fully solvable end-to-end and that the verfiers accept the correct plays.