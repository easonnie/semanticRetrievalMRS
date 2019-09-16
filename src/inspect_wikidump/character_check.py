if __name__ == '__main__':
    # print("Vennela 1 1/2" == "Vennela 1 1/2")
    # print(" 1/2" == " 1/2")
    print(" ".encode('utf-8') == " ".encode('utf-8'))

    print(" ".encode('utf-8'))
    print(" ".encode('utf-8'))  # non-breaking space

    # In the title, it is space, but in text, it is non-breaking space

    # print("1/2" == "1/2")

    from urllib.parse import unquote
    import urllib.parse

    print(urllib.parse.unquote("a%20b") == "a b")
    print(urllib.parse.unquote("a%20b") == "a b")   # non-breaking space not matched.

    # print(" ".encode('utf-8'))
    # print(ord(' ') == 0x202F)
    # print(0x202F)