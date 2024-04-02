# https://stackoverflow.com/questions/63427607/python-upload-files-directly-to-github-using-pygithub


from github import Github


def add_or_update_in_git(access_tocken, github_repo, git_branch, initial_file, folder_empl_in_git):

    g = Github(access_tocken)

    # repo = g.get_user().get_repo(github_repo)
    repo = g.get_repo(github_repo)
    print(repo)

    all_files = []
    contents = repo.get_contents("")

    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file = file_content
            all_files.append(str(file).replace('ContentFile(path="', '').replace('")', ''))

    with open(initial_file, 'r') as file:
        content = file.read()

    # Upload to github
    if folder_empl_in_git in all_files:
        contents = repo.get_contents(folder_empl_in_git)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch=git_branch)
        return folder_empl_in_git + ' UPDATED'
    else:
        repo.create_file(folder_empl_in_git, "committing files", content, branch=git_branch)
        return folder_empl_in_git + ' CREATED'


if __name__ == "__main__":
    ACCESS_TOKEN = input("github personal access token:")
    GIT_BRANCH = "master"

    ACCESS_TOKEN = ""
    GITHUB_REPO = "srio/shadow4tests"
    FOLDER_EMPL_IN_GIT = "workspaces/untitled.ows"

    INTERNAL_FILE = "/users/srio/Oasys/untitled.ows"


    txt = add_or_update_in_git(ACCESS_TOKEN, GITHUB_REPO, GIT_BRANCH, INTERNAL_FILE, FOLDER_EMPL_IN_GIT)
    print(txt)