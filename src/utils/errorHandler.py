def errorHandler(errors):
  errorMessage = []
  for error in errors:
    errorMessage.append({
      'field':error['loc'][0],
      'message':error['msg']
    })
  return errorMessage