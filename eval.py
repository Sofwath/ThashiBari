import turicreate as tc

# load model
model = tc.load_model('thashibarimodel.model')

# load test data
test = tc.load_images('test_images')

# evaluate model
predictions = model.predict(test)
test['predicted_image']= tc.object_detector.util.draw_bounding_boxes(test['image'],predictions)
test[['image', 'predicted_image']].explore()