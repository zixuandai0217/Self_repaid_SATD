-- Query records in SATD table where resolution = SATD_REMOVED, join with SATD_ADDED records to get commit and file information
-- This is the information when SATDs that will eventually be removed are introduced
SELECT DISTINCT
    s_added.satd_id AS satd_id, -- Unique identifier for SATD record, representing a SATD operation (ID of the introduction operation)
    p.p_name AS project_name, -- Project name, obtained from Projects table
    s_added.second_commit AS add_commit_hash, -- Commit hash when SATD_ADDED, representing the commit when SATD was added
    c_added.author_name AS adder_name, -- Author name of the commit when SATD_ADDED
    c_added.author_email AS adder_email, -- Author email of the commit when SATD_ADDED
    c_added.author_date AS add_date, -- Commit date when SATD_ADDED
	c_removed.author_date AS remove_date, -- Commit date when SATD_REMOVED
    f_add.*, -- All fields from SATDInFile table, containing metadata of the file where SATD is located (e.g., file name, class name, method name, line number, etc.)
    CASE 
        WHEN c_added.author_name = c_removed.author_name OR c_added.author_email = c_removed.author_email 
        THEN 1 -- If the author name or email of SATD_ADDED and SATD_REMOVED are the same, mark as self-fixed (1)
        ELSE 0 -- Otherwise mark as non-self-fixed (0)
    END AS is_self_fixed -- Flag indicating whether it is self-fixed, 1 means self-fixed, 0 means non-self-fixed
FROM 
    satd.SATD s_removed -- Main table, SATD table, filter records where resolution = SATD_REMOVED, representing SATD removal operation
JOIN 
    satd.SATD s_added 
    ON s_removed.satd_instance_id = s_added.satd_instance_id -- Join SATD_ADDED record, match by satd_instance_id, satd_instance_id is the unique identifier for the same SATD instance
    AND s_added.resolution = 'SATD_ADDED' -- Ensure the joined record is a SATD addition operation
JOIN 
    satd.Commits c_added 
    ON s_added.p_id = c_added.p_id -- Join commit record of SATD_ADDED by project ID
    AND s_added.second_commit = c_added.commit_hash -- Match commit information of SATD_ADDED by commit hash
JOIN 
    satd.Commits c_removed 
    ON s_removed.p_id = c_removed.p_id -- Join commit record of SATD_REMOVED by project ID
    AND s_removed.second_commit = c_removed.commit_hash -- Match commit information of SATD_REMOVED by commit hash
JOIN 
    satd.SATDInFile f_add 
    ON s_added.second_file = f_add.f_id -- Join file information of SATD_ADDED by second_file (file ID), f_id is the file identifier in SATDInFile table
JOIN 
    satd.Projects p 
    ON s_removed.p_id = p.p_id -- Join project information by project ID to get project name
WHERE 
    s_removed.resolution = 'SATD_REMOVED'; -- Filter condition, ensure only query records of SATD removal operations