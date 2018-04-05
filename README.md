# Aglaia

## Installation notes:
Use the development version of osprey (1.2) by doing:


## Usual problems:
If you get this error:

`
sqlalchemy.exc.StatementError: (builtins.TypeError) Object of type 'ndarray' is not JSON serializable [SQL: 'INSERT INTO trials_v3 (project_name, status, parameters, mean_test_score, mean_train_score, train_scores, test_scores, n_train_samples, n_test_samples, started, completed, elapsed, host, user, traceback, config_sha1) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'] [parameters: [{'started': datetime.datetime(2018, 3, 15, 14, 58, 37, 215783), 'parameters': {'compounds': array([<qml.compound.Compound object at 0x1163ff8d0>,
       <qml.compound.Compound object at 0x1163ffcf8>,
       <qml.compound.Compoun ... (7322 characters truncated) ... epositories/Aglaia/other/osprey_example/tensorboard', 'hl1': 30, 'hl2': 0, 'hl3': 0, 'descriptor': None, 'nuclear_charges': None, 'coordinates': None}, 'status': 'PENDING', 'user': 'walfits', 'config_sha1': '62c64de48fc787c5cbabca64c040e8c75d2b2352', 'project_name': 'default', 'host': 'eduroam-92-45.nomadic.bris.ac.uk', 'mean_train_score': None, 'test_scores': None, 'train_scores': None, 'traceback': None, 'completed': None, 'mean_test_score': None, 'n_test_samples': None, 'n_train_samples': None, 'elapsed': None}]]
`

You probably have forgot to use the development version of Osprey

 
