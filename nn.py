hidden_layer_sizes_bunch = [(64, 64), (64, 32)]
activation_bunch = ["relu"]
solver_bunch = ["adam"]

i = 0
for solver_fun in solver_bunch:
  print("$$$$$$$$$$$\nSOLVER fun")
  print(f"solver_fun  {solver_fun}")
  for activation_fun in activation_bunch:
    print("############")
    print(f"activation_fun  {activation_fun}")
    for layer in hidden_layer_sizes_bunch:
      print(f"layer  {layer}")

      kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      indexes = kf.split(x_train, y_train)
      fin_conf_mat = np. zeros((len(np.unique(y_train)),len(np.unique(y_train))))
      for train_index, test_index in indexes:
          print(i)
          i+=1
          classifier = MLPClassifier(hidden_layer_sizes= layer, activation=activation_fun,
                                    solver=solver_fun, batch_size=50, learning_rate='adaptive', 
                                    learning_rate_init=0.001, max_iter=100, shuffle=True,
                                    random_state=42, early_stopping=True, n_iter_no_change=10,
                                    validation_fraction=0.1, verbose=False)
          classifier.fit(x_train[train_index], y_train[train_index])
          y_pred = classifier.predict(x_train[test_index])
          fin_conf_mat += confusion_matrix(y_train[test_index], y_pred)
      print('konacna matrica konfuzije: \n')
      disp = ConfusionMatrixDisplay(confusion_matrix =fin_conf_mat,  display_labels=classifier.classes_)
      cm = confusion_matrix(np.arange(25), np.arange(25))
      fig, ax = plt.subplots(figsize=(10,10))
      disp.plot(ax=ax, cmap="Blues", values_format='')
      plt.show()

      print('procenat tacno predvidjenih: ', sum(np.diag(fin_conf_mat))/sum(sum(fin_conf_mat)))
      print('prosecna osetljivost je: ', osetljivost_po_klasi(fin_conf_mat, range(10)))