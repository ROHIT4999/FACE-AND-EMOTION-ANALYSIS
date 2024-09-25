-- Create the database
CREATE DATABASE IF NOT EXISTS facerecognition;

-- Use the database
USE facerecognition;

-- Create the table to store user information, emotions, and time of detection
CREATE TABLE IF NOT EXISTS user (
    id INT AUTO_INCREMENT PRIMARY KEY,          -- Unique ID for each entry
    username VARCHAR(100) NOT NULL,             -- Name of the detected user
    emotion VARCHAR(50) NOT NULL,               -- Detected emotion
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP    -- Timestamp of detection
);

-- Optional: Add an index to the username for faster lookups
CREATE INDEX idx_username ON user (username);
