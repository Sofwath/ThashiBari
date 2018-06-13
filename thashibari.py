import turicreate as tc

# define the training and test data
annotations = tc.SFrame('annotations.csv')
images = tc.load_images('training_images')
data = images.join(annotations)
train, test = data.random_split(0.8)

# train and evaluate the model
model = tc.object_detector.create(train)
metrics = model.evaluate(test)

# save the model and export to core ml (to be used in ios)
model.save('thashibarimodel.model')
model.export_coreml('thashibarimodel.mlmodel')