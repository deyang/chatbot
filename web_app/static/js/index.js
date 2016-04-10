$("#say_button").click(function() {
  var txt = $("#msg_input")[0].value;
  $.ajax(
    {
        url: "/message",
        type: "POST",
        dataType: "json",
        success: function(result){
            $("#chat_list").append("<li> You: " + txt + "</li>");
            $("#chat_list").append("<li> Buddy bot: " + result['response'] + "</li>");
        },
        data: {
            "msg": txt
        },
    }
  );
});