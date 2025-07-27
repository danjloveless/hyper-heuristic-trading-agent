-- scripts/init-db.sql
-- Simple database initialization for Docker container

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS quantumtrade;

-- Use the database
USE quantumtrade;

-- Create a simple test table to verify everything works
CREATE TABLE IF NOT EXISTS test_connection (
    id UInt32,
    timestamp DateTime DEFAULT now(),
    message String
) ENGINE = MergeTree()
ORDER BY id;

-- Insert a test record
INSERT INTO test_connection (id, message) VALUES (1, 'ClickHouse is working!');