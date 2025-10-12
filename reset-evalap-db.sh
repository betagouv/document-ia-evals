cd evalap;
docker exec -it postgres-evalap psql -U postgres -d evalap_dev -c "DELETE FROM experiments; DELETE FROM experiment_sets; DELETE FROM datasets; SELECT 'experiments' as table_name, COUNT(*) as count FROM experiments UNION ALL SELECT 'experiment_sets', COUNT(*) FROM experiment_sets UNION ALL SELECT 'datasets', COUNT(*) FROM datasets;";
docker exec -it postgres-evalap psql -U postgres -d evalap_dev -c "DELETE FROM observation_table; DELETE FROM results; DELETE FROM answers; DELETE FROM experiments; DELETE FROM experiment_sets; DELETE FROM datasets;";
